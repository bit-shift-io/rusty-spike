use crate::model::Layer;
use crate::training_metrics::TrainingMetrics;
use rayon::prelude::*;

/// Spike-timing-dependent plasticity (STDP) parameters.
pub struct STDP {
    /// Amplitude of potentiation (pre-before-post)
    pub a_plus: f64,
    /// Amplitude of depression (post-before-pre)
    pub a_minus: f64,
    /// Time constant for potentiation (seconds)
    pub tau_plus: f64,
    /// Time constant for depression (seconds)
    pub tau_minus: f64,
    /// Maximum weight limit
    pub w_max: f64,
    /// Minimum weight limit
    pub w_min: f64,
}

impl STDP {
    pub fn new(
        a_plus: f64,
        a_minus: f64,
        tau_plus: f64,
        tau_minus: f64,
        w_max: f64,
        w_min: f64,
    ) -> Self {
        Self {
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            w_max,
            w_min,
        }
    }

    /// Updates the weights of a model based on STDP.
    ///
    /// - `model`: The model to update.
    /// - `output_spikes`: A slice indicating which neurons in the model spiked this step.
    /// - `input_spikes`: A slice indicating which input channels spiked this step.
    /// - `current_time`: Current simulation time.
    ///
    /// Soft-Bound STDP
    /// Replaced linear STDP updates with a soft-bound rule. Weight changes now scale
    /// with the distance to the boundary (w_max - w for potentiation, w - w_min for depression).
    /// This ensures stable weight convergence without needing aggressive global normalization.
    ///
    pub fn update(
        &self,
        layer: &mut Layer,
        layer_idx: usize,
        output_spikes: &[bool],
        input_spikes: &[bool],
        current_time: f64,
        metrics: &mut TrainingMetrics,
    ) {
        let last_input_spike_times = &layer.last_input_spike_times;

        let results: Vec<(usize, usize, f64, f64)> = layer
            .neurons
            .par_iter_mut()
            .zip(layer.weights.par_iter_mut())
            .enumerate()
            .map(|(i, (neuron, neuron_weights))| {
                let mut local_increased = 0;
                let mut local_decreased = 0;
                let mut local_total_increase = 0.0;
                let mut local_total_decrease = 0.0;

                // 1. Post-synaptic spike: Potentiate weights from inputs that spiked recently
                if output_spikes[i] {
                    for j in 0..last_input_spike_times.len() {
                        if let Some(t_pre) = last_input_spike_times[j] {
                            if t_pre <= current_time {
                                let dt = current_time - t_pre;
                                // Soft-bound potentiation: dw = a_plus * exp(-dt/tau) * (w_max - w)
                                let dw = self.a_plus
                                    * (-dt / self.tau_plus).exp()
                                    * (self.w_max - neuron_weights[j]);
                                let old_weight = neuron_weights[j];
                                let new_weight = (old_weight + dw).min(self.w_max);
                                neuron_weights[j] = new_weight;

                                let actual_dw = new_weight - old_weight;
                                if actual_dw > 0.0 {
                                    local_increased += 1;
                                    local_total_increase += actual_dw;
                                }
                            }
                        }
                    }
                }

                // 2. Pre-synaptic spike: Depress weights to neurons that spiked recently
                if let Some(t_post) = neuron.last_spike_time {
                    if t_post <= current_time {
                        let dt = current_time - t_post;
                        let decay = (-dt / self.tau_minus).exp();
                        for (j, &pre_spiked) in input_spikes.iter().enumerate() {
                            if pre_spiked {
                                // Soft-bound depression: dw = a_minus * decay * (neuron_weights[j] - self.w_min)
                                let dw = self.a_minus * decay * (neuron_weights[j] - self.w_min);
                                let old_weight = neuron_weights[j];
                                let new_weight = (old_weight - dw).max(self.w_min);
                                neuron_weights[j] = new_weight;

                                let actual_dw = new_weight - old_weight;
                                if actual_dw < 0.0 {
                                    local_decreased += 1;
                                    local_total_decrease += actual_dw.abs();
                                }
                            }
                        }
                    }
                }

                (
                    local_increased,
                    local_decreased,
                    local_total_increase,
                    local_total_decrease,
                )
            })
            .collect();

        if let Some(m) = metrics.layers.get_mut(layer_idx) {
            for (inc, dec, inc_val, dec_val) in results {
                m.weights_increased += inc;
                m.weights_decreased += dec;
                m.total_increase += inc_val;
                m.total_decrease += dec_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::Layer, neuron::LIFNeuron};

    #[test]
    fn test_stdp_potentiation() {
        let mut layer = Layer::new(1, 1, LIFNeuron::default());
        let stdp = STDP::new(0.1, 0.1, 0.02, 0.02, 1.0, 0.0);
        let mut metrics = TrainingMetrics::new(&[1usize]);

        // Simulate a pre-synaptic spike at t=0.01
        layer.last_input_spike_times[0] = Some(0.01);

        // Post-synaptic spike at t=0.02
        stdp.update(&mut layer, 0, &[true], &[false], 0.02, &mut metrics);

        // Weight should have increased
        assert!(layer.weights[0][0] > 0.0);
    }

    #[test]
    fn test_stdp_depression() {
        let mut layer = Layer::new(1, 1, LIFNeuron::default());
        let stdp = STDP::new(0.1, 0.1, 0.02, 0.02, 1.0, 0.0);
        let mut metrics = TrainingMetrics::new(&[1usize]);
        layer.weights[0][0] = 0.5;

        // Post-synaptic spike at t=0.01
        layer.neurons[0].last_spike_time = Some(0.01);

        // Pre-synaptic spike at t=0.02
        stdp.update(&mut layer, 0, &[false], &[true], 0.02, &mut metrics);

        // Weight should have decreased
        assert!(layer.weights[0][0] < 0.5);
    }

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new(&[2usize]);
        metrics.record(0, &[true, false]);
        metrics.record(0, &[false, false]);
        metrics.record(0, &[true, true]);
        metrics.add_step();
        metrics.add_step();
        metrics.add_step();

        assert_eq!(metrics.layers[0].firing_counts[0], 2);
        assert_eq!(metrics.layers[0].firing_counts[1], 1);
        assert_eq!(metrics.layers[0].ever_fired.len(), 2);
        assert_eq!(metrics.total_steps, 3);

        metrics.reset_epoch();
        assert_eq!(metrics.layers[0].firing_counts[0], 0);
        assert_eq!(metrics.total_steps, 0);
        assert_eq!(metrics.layers[0].ever_fired.len(), 2); // Coverage persists
    }
}
