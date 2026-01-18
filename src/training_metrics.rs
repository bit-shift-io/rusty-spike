use std::collections::HashSet;

pub struct LayerMetrics {
    pub firing_counts: Vec<usize>,
    pub ever_fired: HashSet<usize>,
    pub weights_increased: usize,
    pub weights_decreased: usize,
    pub total_increase: f64,
    pub total_decrease: f64,
    pub num_neurons: usize,
}

impl LayerMetrics {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            firing_counts: vec![0; num_neurons],
            ever_fired: HashSet::new(),
            weights_increased: 0,
            weights_decreased: 0,
            total_increase: 0.0,
            total_decrease: 0.0,
            num_neurons,
        }
    }

    pub fn reset_epoch(&mut self) {
        self.firing_counts.fill(0);
        self.weights_increased = 0;
        self.weights_decreased = 0;
        self.total_increase = 0.0;
        self.total_decrease = 0.0;
    }
}

pub struct TrainingMetrics {
    pub layers: Vec<LayerMetrics>,
    pub total_steps: usize,
}

impl TrainingMetrics {
    pub fn new(layer_sizes: &[usize]) -> Self {
        Self {
            layers: layer_sizes.iter().map(|&n| LayerMetrics::new(n)).collect(),
            total_steps: 0,
        }
    }

    pub fn record(&mut self, layer_idx: usize, spikes: &[bool]) {
        if layer_idx < self.layers.len() {
            let layer = &mut self.layers[layer_idx];
            for (i, &spiked) in spikes.iter().enumerate() {
                if spiked && i < layer.num_neurons {
                    layer.firing_counts[i] += 1;
                    layer.ever_fired.insert(i);
                }
            }
        }
        // Increment total_steps only once per overall step call (managed externally usually, but we'll stick to per-pattern logic)
    }

    pub fn add_step(&mut self) {
        self.total_steps += 1;
    }

    pub fn report(&self, epoch: usize, dt: f64) {
        let sim_time = self.total_steps as f64 * dt;
        println!(
            "\n  Epoch {} Metrics Summary (Sim Time: {:.3}s):",
            epoch, sim_time
        );
        println!("  {:-<138}", "");
        println!(
            "  {: <6} | {: <10} | {: <12} | {: <10} | {: <10} | {: <12} | {: <12} | {: <12} | {: <12}",
            "Layer",
            "Coverage",
            "Avg Rate",
            "Spikes",
            "Max Rate",
            "W+ Count",
            "W- Count",
            "Avg Inc",
            "Avg Dec"
        );
        println!("  {:-<138}", "");

        for (idx, layer) in self.layers.iter().enumerate() {
            let coverage = (layer.ever_fired.len() as f64 / layer.num_neurons as f64) * 100.0;
            let total_layer_spikes: usize = layer.firing_counts.iter().sum();
            let avg_rate = total_layer_spikes as f64 / (layer.num_neurons as f64 * sim_time);
            let max_spikes = *layer.firing_counts.iter().max().unwrap_or(&0);
            let max_rate = max_spikes as f64 / sim_time;

            let avg_inc = if layer.weights_increased > 0 {
                layer.total_increase / layer.weights_increased as f64
            } else {
                0.0
            };
            let avg_dec = if layer.weights_decreased > 0 {
                layer.total_decrease / layer.weights_decreased as f64
            } else {
                0.0
            };

            println!(
                "  {: <6} | {: >9.1}% | {: >10.2}Hz | {: >10} | {: >8.2}Hz | {: >12} | {: >12} | {: >12.6} | {: >12.6}",
                idx,
                coverage,
                avg_rate,
                total_layer_spikes,
                max_rate,
                layer.weights_increased,
                layer.weights_decreased,
                avg_inc,
                avg_dec
            );
        }
        println!("  {:-<138}\n", "");
    }

    pub fn reset_epoch(&mut self) {
        for layer in &mut self.layers {
            layer.reset_epoch();
        }
        self.total_steps = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new(&[2, 3]);
        metrics.record(0, &[true, false]);
        metrics.record(1, &[true, false, true]);
        metrics.add_step();

        assert_eq!(metrics.layers[0].firing_counts[0], 1);
        assert_eq!(metrics.layers[1].firing_counts[0], 1);
        assert_eq!(metrics.layers[1].firing_counts[2], 1);
        assert_eq!(metrics.total_steps, 1);

        metrics.reset_epoch();
        assert_eq!(metrics.layers[0].firing_counts[0], 0);
        assert_eq!(metrics.total_steps, 0);
    }
}
