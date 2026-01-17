mod data_loader;
mod encoding;
mod labeling;
mod model;
mod neuron;
mod training;
mod training_metrics;

use data_loader::MnistLoader;
use encoding::{Encoder, RateEncoder};
use labeling::Labeler;
use model::Model;
use training::STDP;

use crate::{neuron::LIFNeuron, training_metrics::TrainingMetrics};

fn main() {
    println!("Rusty Spike SNN Simulation");

    // 1. Load MNIST Data (Balanced subset and downsampled for speed)
    println!("Loading MNIST data...");
    let subset_size = 2000;
    let resolution = 14;
    let mut train_set = MnistLoader::load_balanced_subset(
        "data/mnist/train-images-idx3-ubyte",
        "data/mnist/train-labels-idx1-ubyte",
        subset_size,
    )
    .expect("Failed to load training data");
    train_set.downsample(resolution);
    train_set.shuffle();

    let mut test_set = MnistLoader::load_balanced_subset(
        "data/mnist/t10k-images-idx3-ubyte",
        "data/mnist/t10k-labels-idx1-ubyte",
        400,
    )
    .expect("Failed to load test data");
    test_set.downsample(resolution);

    println!(
        "Loaded {} training samples and {} test samples",
        train_set.len(),
        test_set.len()
    );

    // 2. Setup Model and Training Params
    let num_inputs = resolution * resolution;
    let num_neurons = 400;

    let neuron_template = LIFNeuron::new(
        -70.0, // v_rest
        -70.0, // v_reset
        -55.0, // v_threshold_base
        0.3,   // theta_plus (Lower to prevent over-sparsification with many neurons)
        0.001, // theta_decay
        0.02,  // tau_m
        200.0, // r
        0.005, // refractory_period
    );
    let mut model = Model::new(num_neurons, num_inputs, neuron_template);

    // Initialize weights randomly but reasonably
    model.randomize_weights(0.05, 0.4);

    let stdp = STDP::new(
        0.01,  // a_plus
        0.012, // a_minus (Lower to allow broader feature integration)
        0.02,  // tau_plus
        0.02,  // tau_minus
        1.0,   // w_max
        0.0,   // w_min
    );

    let dt = 0.001; // 1ms
    let sim_time = 0.35; // 350ms per pattern
    let steps = (sim_time / dt) as usize;
    let mut encoders = (0..num_inputs)
        .map(|_| RateEncoder::new(300.0)) // Moderate gain
        .collect::<Vec<_>>();

    // 3. Training Loop
    println!("Training...");
    let mut metrics = TrainingMetrics::new(num_neurons);
    model.reset_all(); // Initial reset to clear theta

    let num_epochs = 15;

    for epoch in 0..num_epochs {
        metrics.reset_epoch();
        for i in 0..train_set.len() {
            let data = &train_set.images[i];
            model.reset();
            // Important: Reset encoders for each new image to ensure deterministic but fair start
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

                let potentials_before = model.neurons.iter().map(|n| n.v).collect::<Vec<_>>();
                let output_spikes = model.step(&input_spikes, dt, current_time);

                // Lateral Inhibition (Winner-Takes-All):
                let filtered_spikes = model.apply_lateral_inhibition(
                    &output_spikes,
                    &potentials_before,
                    current_time,
                );

                stdp.update(
                    &mut model,
                    &filtered_spikes,
                    &input_spikes,
                    current_time,
                    &mut metrics,
                );
                // Enforce weight normalization to maintain competition
                metrics.record(&filtered_spikes);
                // Update metrics
                metrics.record(&filtered_spikes);
                current_time += dt;
            }
            // Final normalization for the pattern
            model.normalise_weights(0.4 * num_inputs as f64);
        }
        train_set.shuffle();

        metrics.report(epoch + 1, dt);
    }

    // 4. Calibration: Map neurons to labels
    println!("Calibrating labels...");
    let mut labeler = Labeler::new(num_neurons);
    let mut neuron_selectivity = vec![vec![0usize; 10]; num_neurons];

    for i in 0..train_set.len() {
        let label = train_set.labels[i] as usize;
        let data = &train_set.images[i];
        model.reset();
        for encoder in encoders.iter_mut() {
            encoder.reset();
        }

        let mut current_time = 0.0;
        let mut spike_counts = vec![0usize; num_neurons];
        for _step in 0..steps {
            let input_spikes: Vec<bool> = data
                .iter()
                .enumerate()
                .map(|(i, &val)| encoders[i].step(val, dt, current_time))
                .collect();

            let potentials_before = model.neurons.iter().map(|n| n.v).collect::<Vec<_>>();
            let output_spikes = model.step(&input_spikes, dt, current_time);

            // WTA during calibration as well
            let filtered_spikes =
                model.apply_lateral_inhibition(&output_spikes, &potentials_before, current_time);

            for (i, &s) in filtered_spikes.iter().enumerate() {
                if s {
                    spike_counts[i] += 1;
                }
            }

            current_time += dt;
        }

        // Record which neuron "won" most for this label
        for j in 0..num_neurons {
            neuron_selectivity[j][label] += spike_counts[j];
        }
    }

    labeler.calibrate(&neuron_selectivity);
    println!("  Neuron labels: {:?}", labeler.neuron_to_label);

    // 5. Evaluation
    println!("Evaluating...");
    let mut correct = 0;
    let evaluation_samples = test_set.len();
    assert!(evaluation_samples > 0);

    for i in 0..test_set.len() {
        let label = test_set.labels[i] as usize;
        let data = &test_set.images[i];

        let mut current_time = 0.0;
        model.reset();
        for encoder in encoders.iter_mut() {
            encoder.reset();
        }

        let mut spike_counts = vec![0usize; num_neurons];

        for _step in 0..steps {
            let input_spikes: Vec<bool> = data
                .iter()
                .enumerate()
                .map(|(i, &val)| encoders[i].step(val, dt, current_time))
                .collect();

            let potentials_before = model.neurons.iter().map(|n| n.v).collect::<Vec<_>>();
            let output_spikes = model.step(&input_spikes, dt, current_time);

            // WTA during evaluation is critical for consistency
            let filtered_spikes =
                model.apply_lateral_inhibition(&output_spikes, &potentials_before, current_time);

            for (j, &s) in filtered_spikes.iter().enumerate() {
                if s {
                    spike_counts[j] += 1;
                }
            }
            current_time += dt;
        }

        let prediction = labeler.predict(&spike_counts);

        if prediction == label {
            correct += 1;
        }
    }

    let accuracy = (correct as f64 / test_set.len() as f64) * 100.0;
    println!("Accuracy: {:.2}%", accuracy);

    // Print weights for inspection
    // model.print_weights();
}
