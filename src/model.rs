use crate::neuron::LIFNeuron;

/// A Model represents a layer (or simple network) of spiking neurons.
/// It manages the neurons and their synaptic connections from external inputs.
pub struct Model {
    /// The neurons in this model
    pub neurons: Vec<LIFNeuron>,
    /// Weights from inputs to each neuron. 
    /// weights[i][j] is the weight from input j to neuron i.
    pub weights: Vec<Vec<f64>>,
    /// Last time each input channel spiked (seconds).
    pub last_input_spike_times: Vec<f64>,
}

impl Model {
    /// Creates a new model with a specified number of neurons and input size.
    /// 
    /// All neurons are initialized with the same default LIF parameters.
    pub fn new(num_neurons: usize, num_inputs: usize) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| {
                LIFNeuron::new(
                    -70.0, // v_rest
                    -70.0, // v_reset
                    -50.0, // v_threshold
                    0.02,  // tau_m
                    10.0,  // r
                    0.005, // refractory_period
                )
            })
            .collect();

        // Initialize weights to 0.0
        let weights = vec![vec![0.0; num_inputs]; num_neurons];
        let last_input_spike_times = vec![-f64::INFINITY; num_inputs];

        Self {
            neurons,
            weights,
            last_input_spike_times,
        }
    }

    /// Sets the weight of a specific synapse.
    pub fn set_weight(&mut self, neuron_idx: usize, input_idx: usize, weight: f64) {
        if neuron_idx < self.weights.len() && input_idx < self.weights[0].len() {
            self.weights[neuron_idx][input_idx] = weight;
        }
    }

    /// Performs a single simulation step.
    /// 
    /// - `input_spikes`: A slice representing which input channels spiked.
    /// - `dt`: Time step (seconds).
    /// - `current_time`: Current simulation time (seconds).
    /// 
    /// Returns a vector of booleans indicating which neurons in the model spiked.
    pub fn step(&mut self, input_spikes: &[bool], dt: f64, current_time: f64) -> Vec<bool> {
        let mut output_spikes = Vec::with_capacity(self.neurons.len());

        // Update input spike times
        for (j, &spiked) in input_spikes.iter().enumerate() {
            if spiked && j < self.last_input_spike_times.len() {
                self.last_input_spike_times[j] = current_time;
            }
        }

        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            // Calculate total input current for this neuron
            // sum(weight_j * pulse_j)
            // For simplicity, a spike at an input channel is treated as a unit pulse current.
            let mut i_ext = 0.0;
            for (j, &spiked) in input_spikes.iter().enumerate() {
                if spiked && j < self.weights[i].len() {
                    i_ext += self.weights[i][j];
                }
            }

            // Update neuron state
            let fired = neuron.step(i_ext, dt, current_time);
            output_spikes.push(fired);
        }

        output_spikes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_initialization() {
        let model = Model::new(5, 10);
        assert_eq!(model.neurons.len(), 5);
        assert_eq!(model.weights.len(), 5);
        assert_eq!(model.weights[0].len(), 10);
    }

    #[test]
    fn test_model_step_no_input() {
        let mut model = Model::new(1, 1);
        let spikes = model.step(&[false], 0.001, 0.001);
        assert!(!spikes[0]);
    }

    #[test]
    fn test_model_spike_propagation() {
        let mut model = Model::new(1, 1);
        // Set a huge weight to ensure firing
        model.set_weight(0, 0, 100.0);
        
        let spikes = model.step(&[true], 0.1, 0.1);
        assert!(spikes[0]);
    }
}
