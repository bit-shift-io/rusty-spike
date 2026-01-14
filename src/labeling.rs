pub struct Labeler {
    pub neuron_to_label: Vec<usize>,
}

impl Labeler {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            neuron_to_label: vec![0; num_neurons],
        }
    }

    /// Calibrates the Labeler based on neuron selectivity.
    /// `selectivity[neuron_idx][label_idx]` is the spike count for that neuron for that label.
    pub fn calibrate(&mut self, selectivity: &[Vec<u32>]) {
        let num_neurons = selectivity.len();
        if num_neurons == 0 {
            return;
        }
        let num_labels = selectivity[0].len();
        if num_labels == 0 {
            return;
        }

        // Each neuron picks its "preferred" label (the one it fired most for)
        self.neuron_to_label = selectivity
            .iter()
            .map(|counts| {
                counts
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, count)| count)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();

        // Check for "unclaimed" labels if we want specialization.
        // For now, let's keep it simple: each neuron just gets its best label.
        // The user mentioned the current code is specific to 2 neurons and labels.
        // The previous code had some "collision" logic.

        /*
        The previous logic for 2 neurons was:
        if n0_pref0 >= n0_pref1 && n1_pref1 >= n1_pref0 {
            vec![0, 1]
        } else if n0_pref1 > n0_pref0 && n1_pref0 > n1_pref1 {
            vec![1, 0]
        } else {
            // Collision! Pick labels based on strongest response
            if n0_pref0 + n1_pref1 >= n0_pref1 + n1_pref0 {
                vec![0, 1]
            } else {
                vec![1, 0]
            }
        }
        */
    }

    pub fn predict(&self, spike_counts: &[u32]) -> usize {
        let winner_neuron = spike_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, count)| count)
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.neuron_to_label[winner_neuron]
    }
}
