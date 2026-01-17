use crate::neuron::LIFNeuron;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A Model represents a layer (or simple network) of spiking neurons.
/// It manages the neurons and their synaptic connections from external inputs.
#[derive(Serialize, Deserialize, Clone)]
pub struct Model {
    /// The neurons in this model
    pub neurons: Vec<LIFNeuron>,
    /// Weights from inputs to each neuron.
    /// weights[i][j] is the weight from input j to neuron i.
    pub weights: Vec<Vec<f64>>,
    /// Last time each input channel spiked (seconds).
    pub last_input_spike_times: Vec<Option<f64>>,
}

impl Model {
    /// Creates a new model with a specified number of neurons and input size.
    ///
    /// All neurons are initialized with the same default LIF parameters.
    pub fn new(num_neurons: usize, num_inputs: usize, neuron_template: LIFNeuron) -> Self {
        let neurons = (0..num_neurons).map(|_| neuron_template.clone()).collect();

        // Initialize weights to 0.0
        let weights = vec![vec![0.0; num_inputs]; num_neurons];
        let last_input_spike_times = vec![None; num_inputs];

        Self {
            neurons,
            weights,
            last_input_spike_times,
        }
    }

    /// Sets the weight of a specific synapse.
    #[allow(dead_code)]
    pub fn set_weight(&mut self, neuron_idx: usize, input_idx: usize, weight: f64) {
        if neuron_idx < self.weights.len() && input_idx < self.weights[0].len() {
            self.weights[neuron_idx][input_idx] = weight;
        }
    }

    pub fn randomize_weights(&mut self, min_weight: f64, max_weight: f64) {
        self.weights.par_iter_mut().for_each(|neuron_weights| {
            for weight in neuron_weights.iter_mut() {
                *weight = rand::random::<f64>() * (max_weight - min_weight) + min_weight;
            }
        });
    }

    pub fn normalise_weights(&mut self, target_sum: f64) {
        self.weights.par_iter_mut().for_each(|neuron_weights| {
            let sum: f64 = neuron_weights.iter().sum();
            if sum > 0.0 {
                let factor = target_sum / sum;
                for weight in neuron_weights.iter_mut() {
                    *weight = (*weight * factor).min(1.0).max(0.0);
                }
            }
        });
    }

    /// Performs a single simulation step.
    ///
    /// - `input_spikes`: A slice representing which input channels spiked.
    /// - `dt`: Time step (seconds).
    /// - `current_time`: Current simulation time (seconds).
    ///
    /// Returns a vector of booleans indicating which neurons in the model spiked.
    pub fn step(&mut self, input_spikes: &[bool], dt: f64, current_time: f64) -> Vec<bool> {
        // Update input spike times
        for (j, &spiked) in input_spikes.iter().enumerate() {
            if spiked && j < self.last_input_spike_times.len() {
                self.last_input_spike_times[j] = Some(current_time);
            }
        }

        self.neurons
            .par_iter_mut()
            .zip(self.weights.par_iter())
            .map(|(neuron, neuron_weights)| {
                // Calculate total input current for this neuron
                let mut i_ext = 0.0;
                for (j, &spiked) in input_spikes.iter().enumerate() {
                    if spiked && j < neuron_weights.len() {
                        i_ext += neuron_weights[j];
                    }
                }

                // Update neuron state. returns true if v >= threshold
                neuron.step(i_ext, dt, current_time)
            })
            .collect()
    }

    /// Applies lateral inhibition (Winner-Takes-All) based on membrane potentials.
    /// Returns the filtered spikes (only the winner's spike is kept).
    pub fn apply_lateral_inhibition(
        &mut self,
        would_fire: &[bool],
        _potentials_before: &[f64],
        current_time: f64,
    ) -> Vec<bool> {
        let num_neurons = self.neurons.len();
        let mut filtered_spikes = vec![false; num_neurons];

        // Find winner among those who would fire
        if let Some(winner) = would_fire
            .iter()
            .enumerate()
            .filter(|&(_, &fired)| fired)
            .max_by(|(i, _), (j, _)| {
                // Use current potentials (which are >= threshold) to pick winner
                self.neurons[*i]
                    .v
                    .partial_cmp(&self.neurons[*j].v)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
        {
            filtered_spikes[winner] = true;

            // Fire winner: reset potential and increase theta
            self.neurons[winner].fire(current_time);

            // Inhibition: Reset membrane potential of others who would have fired (or all others?)
            // Conventional SNN: reset all others to rest/reset to enforce competitive sparsity.
            for (i, neuron) in self.neurons.iter_mut().enumerate() {
                if i != winner {
                    neuron.v = neuron.v_reset;
                }
            }
        }

        filtered_spikes
    }

    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
    }

    pub fn reset_all(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset_all();
        }
    }

    // pub fn neuron_idx_with_highest_membrane_potential(&self) -> usize {
    //     self.neurons
    //         .iter()
    //         .enumerate()
    //         .max_by(|(_, a), (_, b)| a.v.partial_cmp(&b.v).unwrap())
    //         .map(|(idx, _)| idx)
    //         .unwrap()
    // }

    #[allow(dead_code)]
    pub fn print_weights(&self) {
        // Print weights for inspection
        println!("Weights:");
        for i in 0..self.neurons.len() {
            print!("  Neuron {}: ", i);
            for j in 0..self.weights[i].len() {
                print!("{:.2} ", self.weights[i][j]);
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_initialization() {
        let model = Model::new(5, 10, LIFNeuron::default());
        assert_eq!(model.neurons.len(), 5);
        assert_eq!(model.weights.len(), 5);
        assert_eq!(model.weights[0].len(), 10);
    }

    #[test]
    fn test_model_step_no_input() {
        let mut model = Model::new(1, 1, LIFNeuron::default());
        let spikes = model.step(&[false], 0.001, 0.001);
        assert!(!spikes[0]);
    }

    #[test]
    fn test_model_spike_propagation() {
        let mut model = Model::new(1, 1, LIFNeuron::default());
        // Set a huge weight to ensure firing
        model.set_weight(0, 0, 100.0);

        let spikes = model.step(&[true], 0.1, 0.1);
        assert!(spikes[0]);
    }
}
