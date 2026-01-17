use std::collections::HashSet;

pub struct TrainingMetrics {
    pub firing_counts: Vec<usize>,
    pub ever_fired: HashSet<usize>,
    pub total_steps: usize,
    pub num_neurons: usize,
    pub weights_increased: usize,
    pub weights_decreased: usize,
    pub total_increase: f64,
    pub total_decrease: f64,
}

impl TrainingMetrics {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            firing_counts: vec![0; num_neurons],
            ever_fired: HashSet::new(),
            total_steps: 0,
            num_neurons,
            weights_increased: 0,
            weights_decreased: 0,
            total_increase: 0.0,
            total_decrease: 0.0,
        }
    }

    pub fn record(&mut self, spikes: &[bool]) {
        self.total_steps += 1;
        for (i, &spiked) in spikes.iter().enumerate() {
            if spiked {
                self.firing_counts[i] += 1;
                self.ever_fired.insert(i);
            }
        }
    }

    pub fn report(&self, epoch: usize, dt: f64) {
        let sim_time = self.total_steps as f64 * dt;
        let coverage = (self.ever_fired.len() as f64 / self.num_neurons as f64) * 100.0;
        println!("  Epoch {} Metrics:", epoch);
        println!(
            "    Neuron Coverage: {:.1}% ({}/{})",
            coverage,
            self.ever_fired.len(),
            self.num_neurons
        );
        for i in 0..self.num_neurons {
            let freq = self.firing_counts[i] as f64 / sim_time;
            println!(
                "    Neuron {}: {} spikes ({:.2} Hz)",
                i, self.firing_counts[i], freq
            );
        }

        // Weight change statistics
        let total_changes = self.weights_increased + self.weights_decreased;
        println!(
            "    Weight Changes: {} increased, {} decreased (total: {})",
            self.weights_increased, self.weights_decreased, total_changes
        );
        println!(
            "    Total Potentiation: {:.4}, Total Depression: {:.4}",
            self.total_increase, self.total_decrease
        );
        if self.weights_increased > 0 {
            println!(
                "    Avg Increase: {:.6}",
                self.total_increase / self.weights_increased as f64
            );
        }
        if self.weights_decreased > 0 {
            println!(
                "    Avg Decrease: {:.6}",
                self.total_decrease / self.weights_decreased as f64
            );
        }
    }

    pub fn reset_epoch(&mut self) {
        self.firing_counts.fill(0);
        self.total_steps = 0;
        self.weights_increased = 0;
        self.weights_decreased = 0;
        self.total_increase = 0.0;
        self.total_decrease = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new(2);
        metrics.record(&[true, false]);
        metrics.record(&[false, false]);
        metrics.record(&[true, true]);

        assert_eq!(metrics.firing_counts[0], 2);
        assert_eq!(metrics.firing_counts[1], 1);
        assert_eq!(metrics.ever_fired.len(), 2);
        assert_eq!(metrics.total_steps, 3);

        metrics.reset_epoch();
        assert_eq!(metrics.firing_counts[0], 0);
        assert_eq!(metrics.total_steps, 0);
        assert_eq!(metrics.ever_fired.len(), 2); // Coverage persists
    }
}
