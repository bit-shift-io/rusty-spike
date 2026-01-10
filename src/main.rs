mod neuron;
mod encoding;
mod model;
mod training;

use encoding::{Encoder, RateEncoder};
use model::Model;
use training::STDP;

fn main() {
    println!("Rusty Spike SNN Simulation");

    // 1. Setup Model and Training Params
    let num_inputs = 10;
    let num_neurons = 2;
    let mut model = Model::new(num_neurons, num_inputs);
    
    // Initialize weights with more variance to encourage differentiation
    for j in 0..num_inputs {
        model.set_weight(0, j, rand::random::<f64>() * 0.3 + 0.1);
        model.set_weight(1, j, rand::random::<f64>() * 0.3 + 0.1);
    }

    let stdp = STDP::new(
        0.15, // a_plus
        0.20, // a_minus
        0.04, // tau_plus
        0.04, // tau_minus
        2.0,  // w_max
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
    let sim_time = 0.3; // 300ms per pattern
    let steps = (sim_time / dt) as usize;
    let mut encoders = (0..num_inputs).map(|_| RateEncoder::new(50.0)).collect::<Vec<_>>();

    // 3. Training Loop
    println!("Training...");
    for epoch in 0..20 {
        for (_label, data) in &patterns {
            let mut current_time = 0.0;
            for _step in 0..steps {
                let input_spikes: Vec<bool> = data.iter().enumerate().map(|(i, &val)| {
                    encoders[i].step(val, dt, current_time)
                }).collect();

                let output_spikes = model.step(&input_spikes, dt, current_time);
                
                let mut filtered_spikes = vec![false; num_neurons];
                if let Some(winner) = output_spikes.iter().position(|&s| s) {
                    filtered_spikes[winner] = true;
                    for (i, neuron) in model.neurons.iter_mut().enumerate() {
                        if i != winner {
                            neuron.v = neuron.v_reset;
                        }
                    }
                }

                stdp.update(&mut model, &filtered_spikes, &input_spikes, current_time);
                current_time += dt;
            }
        }
        if (epoch + 1) % 5 == 0 {
            println!("  Epoch {} complete", epoch + 1);
        }
    }

    // 4. Calibration: Map neurons to labels
    println!("Calibrating labels...");
    let mut neuron_selectivity = vec![vec![0; patterns.len()]; num_neurons];
    for (label, data) in &patterns {
        let mut current_time = 0.0;
        for _step in 0..steps {
            let input_spikes: Vec<bool> = data.iter().enumerate().map(|(i, &val)| {
                encoders[i].step(val, dt, current_time)
            }).collect();
            let output_spikes = model.step(&input_spikes, dt, current_time);
            if let Some(winner) = output_spikes.iter().position(|&s| s) {
                neuron_selectivity[winner][*label] += 1;
                for (i, neuron) in model.neurons.iter_mut().enumerate() {
                    if i != winner { neuron.v = neuron.v_reset; }
                }
            }
            current_time += dt;
        }
    }
    
    let neuron_to_label: Vec<usize> = neuron_selectivity.iter().map(|counts| {
        if counts[0] >= counts[1] { 0 } else { 1 }
    }).collect();
    
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
        
        for _step in 0..steps {
            let input_spikes: Vec<bool> = data.iter().enumerate().map(|(i, &val)| {
                encoders[i].step(val, dt, current_time)
            }).collect();

            let output_spikes = model.step(&input_spikes, dt, current_time);
            if let Some(winner) = output_spikes.iter().position(|&s| s) {
                spike_counts[winner] += 1;
                for (i, neuron) in model.neurons.iter_mut().enumerate() {
                    if i != winner { neuron.v = neuron.v_reset; }
                }
            }
            current_time += dt;
        }

        // Prediction: find which tuned neuron fired most
        let prediction_neuron = if spike_counts[0] > spike_counts[1] { 0 } else { 1 };
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
