use crate::neuron::LIFNeuron;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A Layer represents a layer of spiking neurons.
/// It manages the neurons and their synaptic connections.
#[derive(Serialize, Deserialize, Clone)]
pub struct Layer {
    /// The neurons in this layer
    pub neurons: Vec<LIFNeuron>,
    /// Weights from inputs to each neuron.
    /// weights[i][j] is the weight from input j to neuron i.
    pub weights: Vec<Vec<f64>>,
    /// Last time each input channel spiked (seconds).
    pub last_input_spike_times: Vec<Option<f64>>,
}

impl Layer {
    /// Creates a new layer with a specified number of neurons and input size.
    pub fn new(num_neurons: usize, num_inputs: usize, neuron_template: LIFNeuron) -> Self {
        let neurons = (0..num_neurons).map(|_| neuron_template.clone()).collect();
        let weights = vec![vec![0.0; num_inputs]; num_neurons];
        let last_input_spike_times = vec![None; num_inputs];

        Self {
            neurons,
            weights,
            last_input_spike_times,
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

    /// Performs a single simulation step for this layer.
    pub fn step(&mut self, input_spikes: &[bool], dt: f64, current_time: f64) -> Vec<bool> {
        // Update input spike times for STDP
        for (j, &spiked) in input_spikes.iter().enumerate() {
            if spiked && j < self.last_input_spike_times.len() {
                self.last_input_spike_times[j] = Some(current_time);
            }
        }

        self.neurons
            .par_iter_mut()
            .zip(self.weights.par_iter())
            .map(|(neuron, neuron_weights)| {
                let mut i_ext = 0.0;
                for (j, &spiked) in input_spikes.iter().enumerate() {
                    if spiked && j < neuron_weights.len() {
                        i_ext += neuron_weights[j];
                    }
                }
                neuron.step(i_ext, dt, current_time)
            })
            .collect()
    }

    pub fn apply_lateral_inhibition(
        &mut self,
        would_fire: &[bool],
        current_time: f64,
    ) -> Vec<bool> {
        let num_neurons = self.neurons.len();
        let mut filtered_spikes = vec![false; num_neurons];

        if let Some(winner) = would_fire
            .iter()
            .enumerate()
            .filter(|&(_, &fired)| fired)
            .max_by(|(i, _), (j, _)| {
                self.neurons[*i]
                    .v
                    .partial_cmp(&self.neurons[*j].v)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
        {
            filtered_spikes[winner] = true;
            self.neurons[winner].fire(current_time);
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
}

/// A Model represents the entire multi-layer network.
#[derive(Serialize, Deserialize, Clone)]
pub struct Model {
    pub layers: Vec<Layer>,
}

impl Model {
    pub fn new(layer_configs: Vec<(usize, usize)>, neuron_template: LIFNeuron) -> Self {
        let layers = layer_configs
            .into_iter()
            .map(|(num_neurons, num_inputs)| {
                Layer::new(num_neurons, num_inputs, neuron_template.clone())
            })
            .collect();

        Self { layers }
    }

    pub fn step(&mut self, input_spikes: &[bool], dt: f64, current_time: f64) -> Vec<Vec<bool>> {
        let mut all_spikes = Vec::with_capacity(self.layers.len());
        let mut current_input = input_spikes.to_vec();

        for layer in &mut self.layers {
            let output_spikes = layer.step(&current_input, dt, current_time);
            let filtered_spikes = layer.apply_lateral_inhibition(&output_spikes, current_time);
            all_spikes.push(filtered_spikes.clone());
            current_input = filtered_spikes;
        }

        all_spikes
    }

    pub fn randomize_weights(&mut self, min_weight: f64, max_weight: f64) {
        for layer in &mut self.layers {
            layer.randomize_weights(min_weight, max_weight);
        }
    }

    pub fn normalise_weights(&mut self, target_sums: &[f64]) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < target_sums.len() {
                layer.normalise_weights(target_sums[i]);
            }
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    pub fn reset_all(&mut self) {
        for layer in &mut self.layers {
            layer.reset_all();
        }
    }

    #[allow(dead_code)]
    pub fn print_weights(&self) {
        for (idx, layer) in self.layers.iter().enumerate() {
            println!("Layer {}:", idx);
            for i in 0..layer.neurons.len() {
                print!("  Neuron {}: ", i);
                for j in 0..layer.weights[i].len() {
                    print!("{:.2} ", layer.weights[i][j]);
                }
                println!();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_initialization() {
        let layer = Layer::new(5, 10, LIFNeuron::default());
        assert_eq!(layer.neurons.len(), 5);
        assert_eq!(layer.weights.len(), 5);
        assert_eq!(layer.weights[0].len(), 10);
    }

    #[test]
    fn test_layer_step_no_input() {
        let mut layer = Layer::new(1, 1, LIFNeuron::default());
        let spikes = layer.step(&[false], 0.001, 0.001);
        assert!(!spikes[0]);
    }

    #[test]
    fn test_model_multi_layer() {
        let model = Model::new(vec![(10, 5), (5, 10)], LIFNeuron::default());
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.layers[0].neurons.len(), 10);
        assert_eq!(model.layers[1].neurons.len(), 5);
    }
}
