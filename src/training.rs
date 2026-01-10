use crate::model::Model;

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
    pub fn update(
        &self,
        model: &mut Model,
        output_spikes: &[bool],
        input_spikes: &[bool],
        current_time: f64,
    ) {
        // 1. Post-synaptic spike: Potentiate weights from inputs that spiked recently
        for (i, &post_spiked) in output_spikes.iter().enumerate() {
            if post_spiked {
                for j in 0..model.last_input_spike_times.len() {
                    let t_pre = model.last_input_spike_times[j];
                    if t_pre <= current_time && t_pre > -f64::INFINITY {
                        let dt = current_time - t_pre;
                        let dw = self.a_plus * (-dt / self.tau_plus).exp();
                        model.weights[i][j] = (model.weights[i][j] + dw).min(self.w_max);
                    }
                }
            }
        }

        // 2. Pre-synaptic spike: Depress weights to neurons that spiked recently
        for (j, &pre_spiked) in input_spikes.iter().enumerate() {
            if pre_spiked {
                for i in 0..model.neurons.len() {
                    let t_post = model.neurons[i].last_spike_time;
                    if t_post <= current_time && t_post > -f64::INFINITY {
                        let dt = current_time - t_post;
                        let dw = self.a_minus * (-dt / self.tau_minus).exp();
                        model.weights[i][j] = (model.weights[i][j] - dw).max(self.w_min);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::Model, neuron::LIFNeuron};

    #[test]
    fn test_stdp_potentiation() {
        let mut model = Model::new(1, 1, LIFNeuron::default());
        let stdp = STDP::new(0.1, 0.1, 0.02, 0.02, 1.0, 0.0);

        // Simulate a pre-synaptic spike at t=0.01
        model.last_input_spike_times[0] = 0.01;

        // Post-synaptic spike at t=0.02
        stdp.update(&mut model, &[true], &[false], 0.02);

        // Weight should have increased
        assert!(model.weights[0][0] > 0.0);
    }

    #[test]
    fn test_stdp_depression() {
        let mut model = Model::new(1, 1, LIFNeuron::default());
        let stdp = STDP::new(0.1, 0.1, 0.02, 0.02, 1.0, 0.0);
        model.weights[0][0] = 0.5;

        // Post-synaptic spike at t=0.01
        model.neurons[0].last_spike_time = 0.01;

        // Pre-synaptic spike at t=0.02
        stdp.update(&mut model, &[false], &[true], 0.02);

        // Weight should have decreased
        assert!(model.weights[0][0] < 0.5);
    }
}
