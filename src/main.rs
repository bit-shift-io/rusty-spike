mod data_loader;
mod encoding;
mod labeling;
mod model;
mod neuron;
mod persistence;
mod training;
mod training_metrics;

use data_loader::MnistLoader;
use encoding::{Encoder, RateEncoder};
use labeling::Labeler;
use model::Model;
use persistence::ModelCheckpoint;
use training::STDP;

use crate::neuron::LIFNeuron;
use crate::training_metrics::TrainingMetrics;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;

fn main() {
    println!("Rusty Spike SNN Simulation");

    // 1. Load MNIST Data (Balanced subset and downsampled for speed)
    println!("Loading MNIST data...");
    let subset_size = 1000; // Increased for better learning
    let resolution = 12;
    let bit_depth = 4; // Increased bit depth for better input precision
    let mut train_set = MnistLoader::load_balanced_subset(
        "data/mnist/train-images-idx3-ubyte",
        "data/mnist/train-labels-idx1-ubyte",
        subset_size,
    )
    .expect("Failed to load training data");
    train_set.downsample(resolution);
    train_set.reduce_bit_depth(bit_depth);
    train_set.shuffle();

    let mut test_set = MnistLoader::load_balanced_subset(
        "data/mnist/t10k-images-idx3-ubyte",
        "data/mnist/t10k-labels-idx1-ubyte",
        400,
    )
    .expect("Failed to load test data");
    test_set.downsample(resolution);
    test_set.reduce_bit_depth(bit_depth);

    println!(
        "Loaded {} training samples and {} test samples (Bit Depth: {})",
        train_set.len(),
        test_set.len(),
        bit_depth
    );

    // 2. Setup Model and Training Params
    let num_inputs = resolution * resolution;
    // Multi-layer config: Layer 1 (256 neurons), Layer 2 (100 neurons)
    // Deepening the network as requested.
    let layer_configs = vec![(256, num_inputs), (100, 256)];
    let layer_sizes: Vec<usize> = layer_configs.iter().map(|(n, _)| *n).collect();

    let last_layer_idx = layer_configs.len() - 1;
    let num_output_neurons = layer_configs[last_layer_idx].0;

    let neuron_template = LIFNeuron::new(
        -70.0, // v_rest
        -70.0, // v_reset
        -55.0, // v_threshold_base
        1.5,   // theta_plus: Increased to force more lateral diversity
        0.05,  // theta_decay: Faster decay to prevent dead neurons
        0.02,  // tau_m
        250.0, // r: Increased resistance for more sensitivity
        0.005, // refractory_period
    );

    let load_existing = false; // Fresh start with new params

    let checkpoint_path = "model_checkpoint.json";
    let (mut model, start_epoch) = if load_existing && Path::new(checkpoint_path).exists() {
        println!("Loading existing model from {}...", checkpoint_path);
        match ModelCheckpoint::load(checkpoint_path) {
            Ok(cp) => {
                let epoch = cp
                    .metadata
                    .get("epoch")
                    .and_then(|e| e.parse::<usize>().ok())
                    .unwrap_or(0);
                println!("Resuming from epoch {}", epoch);
                (cp.model, epoch)
            }
            Err(e) => {
                println!("Warning: Failed to load checkpoint: {}. Starting fresh.", e);
                let mut fresh_model = Model::new(layer_configs, neuron_template);
                fresh_model.randomize_weights(0.05, 0.4);
                (fresh_model, 0)
            }
        }
    } else {
        println!("No checkpoint found. Initializing new model.");
        let mut fresh_model = Model::new(layer_configs, neuron_template);
        fresh_model.randomize_weights(0.05, 0.4);
        (fresh_model, 0)
    };

    let initial_a_plus = 0.02; // Increased slightly
    let initial_a_minus = 0.025;

    let dt = 0.001; // 1ms
    let sim_time = 0.15; // 150ms: Increased to allow more spikes per pattern
    let steps = (sim_time / dt) as usize;
    let mut encoders = (0..num_inputs)
        .map(|_| RateEncoder::new(150.0)) // Increased gain to boost firing
        .collect::<Vec<_>>();

    // 3. Training Loop
    println!("Training...");
    // Use all layers for metrics
    let mut metrics = TrainingMetrics::new(&layer_sizes);
    model.reset_all();

    let num_epochs = 10; // Increased epochs for better learning

    for epoch in start_epoch..(start_epoch + num_epochs) {
        // Gentler learning rate decay
        let decay = 0.7_f64.powf((epoch - start_epoch) as f64 / num_epochs as f64);
        let current_a_plus = initial_a_plus * decay;
        let current_a_minus = initial_a_minus * decay;

        let stdp = STDP::new(
            current_a_plus,
            current_a_minus,
            0.02, // tau_plus
            0.02, // tau_minus
            1.0,  // w_max
            0.0,  // w_min
        );

        print!(
            "Epoch {} (a_plus: {:.4}, a_minus: {:.4})",
            (epoch + 1),
            current_a_plus,
            current_a_minus
        );
        io::stdout().flush().unwrap();
        metrics.reset_epoch();

        for i in 0..train_set.len() {
            if i % 100 == 0 {
                print!(".");
                io::stdout().flush().unwrap();
            }

            let data = &train_set.images[i];
            model.reset();
            for encoder in encoders.iter_mut() {
                encoder.reset();
            }

            let mut current_time = 0.0;
            for _step in 0..steps {
                let input_spikes: Vec<bool> = data
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| encoders[i].step(val, dt, current_time))
                    .collect();

                // Propagation through all layers
                let all_spikes = model.step(&input_spikes, dt, current_time);

                // Update each layer with STDP
                // Layer 0: inputs are image spikes, outputs are all_spikes[0]
                stdp.update(
                    &mut model.layers[0],
                    0,
                    &all_spikes[0],
                    &input_spikes,
                    current_time,
                    &mut metrics,
                );

                // Layer 1..n: inputs are all_spikes[i-1], outputs are all_spikes[i]
                for l in 1..model.layers.len() {
                    let (prev_spikes, current_layer_spikes) = (&all_spikes[l - 1], &all_spikes[l]);
                    stdp.update(
                        &mut model.layers[l],
                        l,
                        current_layer_spikes,
                        prev_spikes,
                        current_time,
                        &mut metrics,
                    );
                }

                for l in 0..model.layers.len() {
                    metrics.record(l, &all_spikes[l]);
                }
                metrics.add_step();
                current_time += dt;
            }

            // Normalisation per pattern
            let target_sums: Vec<f64> = model
                .layers
                .iter()
                .enumerate()
                .map(|(l, _layer)| {
                    let inputs_to_this_layer = if l == 0 {
                        num_inputs
                    } else {
                        model.layers[l - 1].neurons.len()
                    };
                    0.25 * inputs_to_this_layer as f64 // Lowered normalization target to encourage sparsity
                })
                .collect();
            model.normalise_weights(&target_sums);
        }
        println!("");
        train_set.shuffle();

        metrics.report(epoch + 1, dt);

        // Save progress after each epoch
        let mut metadata = HashMap::new();
        metadata.insert("epoch".to_string(), (epoch + 1).to_string());
        metadata.insert("num_layers".to_string(), model.layers.len().to_string());
        metadata.insert("resolution".to_string(), resolution.to_string());

        let cp = ModelCheckpoint::new(model.clone(), metadata);
        if let Err(e) = cp.save(checkpoint_path) {
            println!("Warning: Failed to save checkpoint: {}", e);
        } else {
            println!("  Checkpoint saved to {}", checkpoint_path);
        }
    }

    // 4. Calibration: Map neurons to labels
    println!("Calibrating labels...");
    let mut labeler = Labeler::new(num_output_neurons);
    let mut neuron_selectivity = vec![vec![0usize; 10]; num_output_neurons];

    let neuron_selectivity_results: Vec<(usize, Vec<usize>)> = (0..train_set.len())
        .into_par_iter()
        .map(|i| {
            let label = train_set.labels[i] as usize;
            let data = &train_set.images[i];
            let mut local_model = model.clone();
            let mut local_encoders = encoders.clone();

            local_model.reset();
            for encoder in local_encoders.iter_mut() {
                encoder.reset();
            }

            let mut current_time = 0.0;
            let mut spike_counts = vec![0usize; num_output_neurons];
            for _step in 0..steps {
                let input_spikes: Vec<bool> = data
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| local_encoders[i].step(val, dt, current_time))
                    .collect();

                let all_spikes = local_model.step(&input_spikes, dt, current_time);
                let last_layer_spikes = &all_spikes[last_layer_idx];

                for (idx, &s) in last_layer_spikes.iter().enumerate() {
                    if s {
                        spike_counts[idx] += 1;
                    }
                }

                current_time += dt;
            }
            (label, spike_counts)
        })
        .collect();

    for (label, spike_counts) in neuron_selectivity_results {
        for j in 0..num_output_neurons {
            neuron_selectivity[j][label] += spike_counts[j];
        }
    }

    labeler.calibrate(&neuron_selectivity);
    println!("  Neuron labels: {:?}", labeler.neuron_to_label);

    // 5. Evaluation
    println!("Evaluating...");
    let evaluation_samples = test_set.len();
    assert!(evaluation_samples > 0);

    let correct: usize = (0..test_set.len())
        .into_par_iter()
        .map(|i| {
            let label = test_set.labels[i] as usize;
            let data = &test_set.images[i];

            let mut current_time = 0.0;
            let mut local_model = model.clone();
            let mut local_encoders = encoders.clone();

            local_model.reset();
            for encoder in local_encoders.iter_mut() {
                encoder.reset();
            }

            let mut spike_counts = vec![0usize; num_output_neurons];

            for _step in 0..steps {
                let input_spikes: Vec<bool> = data
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| local_encoders[i].step(val, dt, current_time))
                    .collect();

                let all_spikes = local_model.step(&input_spikes, dt, current_time);
                let last_layer_spikes = &all_spikes[last_layer_idx];

                for (j, &s) in last_layer_spikes.iter().enumerate() {
                    if s {
                        spike_counts[j] += 1;
                    }
                }
                current_time += dt;
            }

            let prediction = labeler.predict(&spike_counts);

            if prediction == label { 1 } else { 0 }
        })
        .sum();

    let accuracy = (correct as f64 / test_set.len() as f64) * 100.0;
    println!("Accuracy: {:.2}%", accuracy);
}
