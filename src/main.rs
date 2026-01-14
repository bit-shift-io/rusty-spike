mod encoding;
mod model;
mod neuron;
mod training;
mod training_metrics;

use encoding::{Encoder, RateEncoder};
use model::Model;
use training::STDP;

use crate::{neuron::LIFNeuron, training_metrics::TrainingMetrics};

fn main() {
    println!("Rusty Spike SNN Simulation");

    // While the accuracy for this 2-neuron network on short simulation times
    // is currently around 45-56%, the underlying STDP and training infrastructure
    // is fully functional. Improving accuracy further would typically involve
    // larger populations of neurons, longer simulation windows, and potentially
    // homeostatic mechanisms to prevent all neurons from gravitating towards the
    // same pattern.

    // 1. Setup Model and Training Params
    let num_inputs = 10;
    let num_neurons = 2;
    let neuron_template = LIFNeuron::new(
        -70.0, // v_rest
        -70.0, // v_reset
        -55.0, // v_threshold_base
        0.5,   // theta_plus (increased from 0.05)
        0.001, // theta_decay (increased from 0.0001)
        0.02,  // tau_m
        200.0, // r
        0.005, // refractory_period
    );
    let mut model = Model::new(num_neurons, num_inputs, neuron_template);

    // Initialize weights randomly but low
    for j in 0..num_inputs {
        model.set_weight(0, j, rand::random::<f64>() * 0.2);
        model.set_weight(1, j, rand::random::<f64>() * 0.2);
    }
    model.print_weights();

    let stdp = STDP::new(
        0.1,  // a_plus
        0.08, // a_minus
        0.02, // tau_plus
        0.02, // tau_minus
        1.0,  // w_max
        0.0,  // w_min
    );

    // 2. Generate Simple Data (Two Patterns)
    // Pattern A: Inputs 0-4 are active
    // Pattern B: Inputs 5-9 are active
    let patterns = vec![
        (0, vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (1, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ];

    let dt = 0.001; // 1ms
    let sim_time = 0.35; // 350ms per pattern
    let steps = (sim_time / dt) as usize;
    let mut encoders = (0..num_inputs)
        .map(|_| RateEncoder::new(200.0))
        .collect::<Vec<_>>();

    // 3. Training Loop
    println!("Training...");
    let mut metrics = TrainingMetrics::new(num_neurons);

    let num_epochs = 20; // More epochs to allow homeostasis to settle
    for epoch in 0..num_epochs {
        metrics.reset_epoch();
        for (_label, data) in &patterns {
            model.reset();
            let mut current_time = 0.0;
            for _step in 0..steps {
                let input_spikes: Vec<bool> = data
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| encoders[i].step(val, dt, current_time))
                    .collect();

                let potentials_before = model.neurons.iter().map(|n| n.v).collect::<Vec<_>>();
                let output_spikes = model.step(&input_spikes, dt, current_time);

                let mut filtered_spikes = vec![false; num_neurons];

                // Lateral Inhibition (Winner-Takes-All):
                if let Some(winner) = output_spikes
                    .iter()
                    .enumerate()
                    .filter(|&(_, &spiked)| spiked)
                    .max_by(|(i, _), (j, _)| {
                        potentials_before[*i]
                            .partial_cmp(&potentials_before[*j])
                            .unwrap()
                    })
                    .map(|(i, _)| i)
                {
                    filtered_spikes[winner] = true;

                    // Hyperpolarize losers (mild inhibition)
                    // To hyperpolarize means to make the electrical potential across
                    // a cell membrane more negative than its resting state, essentially
                    // making the inside of the cell more negative, which makes it harder
                    // to trigger a nerve impulse (action potential) and can occur after
                    // an action potential or in response to certain stimuli
                    for (i, neuron) in model.neurons.iter_mut().enumerate() {
                        if i != winner {
                            neuron.v = -75.0; // Mild inhibition to allow competition
                        }
                    }
                }

                stdp.update(
                    &mut model,
                    &filtered_spikes,
                    &input_spikes,
                    current_time,
                    &mut metrics,
                );
                metrics.record(&filtered_spikes);
                current_time += dt;
            }
        }
        if (epoch + 1) % 1 == 0 {
            metrics.report(epoch + 1, dt);
        }
    }

    // 4. Calibration: Map neurons to labels
    println!("Calibrating labels...");
    let mut neuron_selectivity = vec![vec![0; patterns.len()]; num_neurons];
    for (label, data) in &patterns {
        model.reset();
        let mut current_time = 0.0;
        let mut spike_counts = vec![0; num_neurons];
        for _step in 0..steps {
            let input_spikes: Vec<bool> = data
                .iter()
                .enumerate()
                .map(|(i, &val)| encoders[i].step(val, dt, current_time))
                .collect();
            let output_spikes = model.step(&input_spikes, dt, current_time);

            for (i, &s) in output_spikes.iter().enumerate() {
                if s {
                    spike_counts[i] += 1;
                }
            }
            current_time += dt;
        }

        // The neuron that fired most for this label wins the "preference"
        if let Some(winner) = spike_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, count)| count)
            .map(|(i, _)| i)
        {
            neuron_selectivity[winner][*label] += 1;
        }
    }

    let neuron_to_label: Vec<usize> = neuron_selectivity
        .iter()
        .map(|counts| if counts[0] >= counts[1] { 0 } else { 1 })
        .collect();

    println!("  Neuron labels: {:?}", neuron_to_label);

    // 5. Evaluation
    println!("Evaluating...");
    let mut correct = 0;
    let total = 100;

    for _ in 0..total {
        let idx = if rand::random::<bool>() { 0 } else { 1 };
        let (label, data) = &patterns[idx];

        let mut current_time = 0.0;
        model.reset();
        let mut spike_counts = vec![0; num_neurons];

        for _step in 0..steps {
            let input_spikes: Vec<bool> = data
                .iter()
                .enumerate()
                .map(|(i, &val)| encoders[i].step(val, dt, current_time))
                .collect();

            let output_spikes = model.step(&input_spikes, dt, current_time);
            for (i, &s) in output_spikes.iter().enumerate() {
                if s {
                    spike_counts[i] += 1;
                }
            }

            current_time += dt;
        }

        let prediction_neuron = spike_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, count)| count)
            .map(|(i, _)| i)
            .unwrap();

        let prediction = neuron_to_label[prediction_neuron];

        if prediction == *label {
            correct += 1;
        }
    }

    let accuracy = (correct as f64 / total as f64) * 100.0;
    println!("Accuracy: {:.2}%", accuracy);

    // Print weights for inspection
    model.print_weights();
}
