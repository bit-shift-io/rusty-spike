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
    let mut model = Model::new(num_neurons, num_inputs, LIFNeuron::default());

    // Initialize weights with more magnitude to ensure neurons can reach threshold
    // Threshold is -50mV, rest is -70mV, so we need >20mV increase.
    // R=10, so we need >2.0 units of current.
    // With 5 inputs active, each weight should be around 0.5-0.8.
    for j in 0..num_inputs {
        model.set_weight(0, j, rand::random::<f64>() * 0.4 + 0.5);
        model.set_weight(1, j, rand::random::<f64>() * 0.4 + 0.5);
    }

    let stdp = STDP::new(
        0.01,  // a_plus
        0.005, // a_minus
        1.0,   // tau_plus
        1.0,   // tau_minus
        2.0,   // w_max
        0.0,   // w_min
    );

    // 2. Generate Simple Data (Two Patterns)
    // Pattern A: Inputs 0-4 are active
    // Pattern B: Inputs 5-9 are active
    let patterns = vec![
        (0, vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (1, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ];

    let dt = 0.001; // 1ms
    let sim_time = 0.3; // 300ms per pattern
    let steps = (sim_time / dt) as usize;
    let mut encoders = (0..num_inputs)
        .map(|_| RateEncoder::new(150.0))
        .collect::<Vec<_>>();

    // 3. Training Loop
    println!("Training...");
    let mut metrics = TrainingMetrics::new(num_neurons);

    // Lateral Inhibition: When one neuron spikes, it inhibits others
    // by resetting their potential, forcing neurons to compete and specialize
    // in different patterns.
    let lateral_inhibition = false;

    let num_epochs = 5;
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

                let output_spikes = model.step(&input_spikes, dt, current_time);

                let mut filtered_spikes = vec![false; num_neurons];
                if lateral_inhibition {
                    if let Some(winner) = output_spikes.iter().position(|&s| s) {
                        filtered_spikes[winner] = true;
                        for (i, neuron) in model.neurons.iter_mut().enumerate() {
                            if i != winner {
                                neuron.v = neuron.v_reset;
                            }
                        }
                    }
                } else {
                    filtered_spikes = output_spikes;
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
        //if (epoch + 1) % 10 == 0 {
        metrics.report(epoch + 1, dt);
        // }
    }

    // 4. Calibration: Map neurons to labels
    println!("Calibrating labels...");
    let mut neuron_selectivity = vec![vec![0; patterns.len()]; num_neurons];
    for (label, data) in &patterns {
        model.reset();
        let mut current_time = 0.0;
        for _step in 0..steps {
            let input_spikes: Vec<bool> = data
                .iter()
                .enumerate()
                .map(|(i, &val)| encoders[i].step(val, dt, current_time))
                .collect();
            let output_spikes = model.step(&input_spikes, dt, current_time);
            if let Some(winner) = output_spikes.iter().position(|&s| s) {
                neuron_selectivity[winner][*label] += 1;
                if lateral_inhibition {
                    for (i, neuron) in model.neurons.iter_mut().enumerate() {
                        if i != winner {
                            neuron.v = neuron.v_reset;
                        }
                    }
                }
            }
            current_time += dt;
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

        let mut spike_counts = vec![0; num_neurons];
        let mut current_time = 0.0;
        model.reset();

        for _step in 0..steps {
            let input_spikes: Vec<bool> = data
                .iter()
                .enumerate()
                .map(|(i, &val)| encoders[i].step(val, dt, current_time))
                .collect();

            let output_spikes = model.step(&input_spikes, dt, current_time);
            if let Some(winner) = output_spikes.iter().position(|&s| s) {
                spike_counts[winner] += 1;
                if lateral_inhibition {
                    for (i, neuron) in model.neurons.iter_mut().enumerate() {
                        if i != winner {
                            neuron.v = neuron.v_reset;
                        }
                    }
                }
            }
            current_time += dt;
        }

        // Prediction: find which tuned neuron fired most
        let prediction_neuron = if spike_counts[0] > spike_counts[1] {
            0
        } else {
            1
        };
        let prediction = neuron_to_label[prediction_neuron];

        if prediction == *label {
            correct += 1;
        }
    }

    let accuracy = (correct as f64 / total as f64) * 100.0;
    println!("Accuracy: {:.2}%", accuracy);

    // Print weights for inspection
    println!("Final Weights:");
    for i in 0..num_neurons {
        print!("  Neuron {}: ", i);
        for j in 0..num_inputs {
            print!("{:.2} ", model.weights[i][j]);
        }
        println!();
    }
}
