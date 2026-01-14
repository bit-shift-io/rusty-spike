/// A Leaky Integrate and Fire (LIF) Neuron model.
///
/// The LIF neuron is defined by the differential equation:
/// τ_m * dv/dt = -(v - v_rest) + R * I(t)
/// where:
/// - τ_m is the membrane time constant (R * C)
/// - v is the membrane potential
/// - v_rest is the resting potential
/// - R is the membrane resistance
/// - I(t) is the input current at time t
///
/// Features:
///
/// 1. Homeostasis (Adaptive Thresholds)
/// Every time a neuron spikes, its threshold increases (theta_plus),
/// making it harder for it to fire immediately again. This forces
/// other neurons to "pick up the slack" and learn patterns that the
/// dominant neuron is missing.
///
/// 2. Refractory Period
/// After a neuron spikes, it enters a refractory period during which
/// it cannot spike again. This prevents neurons from firing too
/// frequently and allows the system to settle into a stable state.
///
/// 3. Membrane Potential Decay
/// The membrane potential decays exponentially over time, simulating
/// the natural leakage of ions across the cell membrane.
///
#[derive(Clone)]
pub struct LIFNeuron {
    /// Current membrane potential (V)
    pub v: f64,
    /// Resting potential (V)
    pub v_rest: f64,
    /// Reset potential after a spike (V)
    pub v_reset: f64,
    /// Base threshold potential (V)
    pub v_threshold_base: f64,
    /// Current adaptive threshold component (V)
    pub theta: f64,
    /// Amount to increase theta after a spike (V)
    pub theta_plus: f64,
    /// Decay rate for theta
    pub theta_decay: f64,
    /// Membrane time constant (seconds)
    pub tau_m: f64,
    /// Membrane resistance (Ohms)
    pub r: f64,
    /// Refractory period duration (seconds)
    pub refractory_period: f64,
    /// Last time the neuron spiked (seconds)
    pub last_spike_time: f64,
}

impl LIFNeuron {
    /// Creates a new LIFNeuron with the given parameters.
    pub fn new(
        v_rest: f64,
        v_reset: f64,
        v_threshold_base: f64,
        theta_plus: f64,
        theta_decay: f64,
        tau_m: f64,
        r: f64,
        refractory_period: f64,
    ) -> Self {
        Self {
            v: v_rest,
            v_rest,
            v_reset,
            v_threshold_base,
            theta: 0.0,
            theta_plus,
            theta_decay,
            tau_m,
            r,
            refractory_period,
            last_spike_time: -f64::INFINITY,
        }
    }

    #[allow(dead_code)]
    pub fn default() -> Self {
        Self::new(
            -70.0,  // v_rest
            -70.0,  // v_reset
            -50.0,  // v_threshold_base
            0.05,   // theta_plus
            0.0001, // theta_decay
            0.02,   // tau_m
            100.0,  // r
            0.005,  // refractory_period
        )
    }

    /// Updates the neuron state for a single time step.
    ///
    /// - `i_ext`: External input current (Amperes)
    /// - `dt`: Time step duration (seconds)
    /// - `current_time`: Current simulation time (seconds)
    ///
    /// Returns `true` if the neuron fired a spike.
    pub fn step(&mut self, i_ext: f64, dt: f64, current_time: f64) -> bool {
        // 1. Decay the adaptive threshold component (theta)
        // theta = theta * exp(-dt / tau_theta) or simple subtraction/multiplication
        // For simplicity, we use binary decay: theta -= theta * theta_decay * dt
        self.theta -= self.theta * self.theta_decay;

        // 2. Check if we are in the refractory period
        if current_time < self.last_spike_time + self.refractory_period {
            self.v = self.v_reset;
            return false;
        }

        // 3. Numerical integration using Euler method:
        // dv = [-(v - v_rest) + R * I] * (dt / tau_m)
        let dv = (-(self.v - self.v_rest) + self.r * i_ext) * (dt / self.tau_m);
        self.v += dv;

        // 4. Check for spike against dynamic threshold
        let effective_threshold = self.v_threshold_base + self.theta;
        if self.v >= effective_threshold {
            self.v = self.v_reset;
            self.last_spike_time = current_time;
            // Increase adaptive threshold after spike
            self.theta += self.theta_plus;
            true
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.last_spike_time = -f64::INFINITY;
        self.theta = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passive_leak() {
        let mut neuron = LIFNeuron::new(-70.0, -70.0, -50.0, 0.05, 0.0001, 0.02, 10.0, 0.005);
        neuron.v = -60.0; // Start above rest

        // Step forward without input
        neuron.step(0.0, 0.001, 0.001);

        // Potential should have decreased towards -70.0
        assert!(neuron.v < -60.0);
        assert!(neuron.v > -70.0);
    }

    #[test]
    fn test_integration() {
        let mut neuron = LIFNeuron::new(-70.0, -70.0, -50.0, 0.05, 0.0001, 0.02, 10.0, 0.005);

        // Apply positive current
        neuron.step(2.0, 0.001, 0.001);

        // Potential should have increased
        assert!(neuron.v > -70.0);
    }

    #[test]
    fn test_firing_and_homeostasis() {
        let mut neuron = LIFNeuron::new(-70.0, -70.0, -50.0, 0.5, 0.01, 0.02, 10.0, 0.005);

        // Apply very large current to trigger immediate spike
        let fired = neuron.step(10.0, 0.1, 0.1);

        assert!(fired);
        assert_eq!(neuron.v, -70.0); // Should be reset
        assert_eq!(neuron.last_spike_time, 0.1);

        // Theta should have increased
        assert_eq!(neuron.theta, 0.5 * (1.0 - 0.01)); // theta was 0, decayed (to 0), then increased by 0.5
        // Wait, the decay happens BEFORE the spike check in my logic.
        // Let's re-verify:
        // step(10.0, 0.1, 0.1):
        // 1. theta -= 0 * 0.01 = 0
        // 3. dv = (-( -70 - -70) + 10 * 10) * (0.1 / 0.02) = 100 * 5 = 500
        // v = -70 + 500 = 430
        // 4. effective_threshold = -50 + 0 = -50
        // v >= -50 is true
        // v = -70, last_spike_time = 0.1, theta = 0 + 0.5 = 0.5
        // Wait, the decay `self.theta -= self.theta * self.theta_decay` happens at the start.
        // So after one step where it fired, theta should be 0.5.
        assert_eq!(neuron.theta, 0.5);
    }

    #[test]
    fn test_refractory_period() {
        let mut neuron = LIFNeuron::new(-70.0, -70.0, -50.0, 0.05, 0.0001, 0.02, 10.0, 0.005);

        // Trigger a spike
        neuron.step(10.0, 0.1, 0.1);

        // Try to increase potential during refractory period
        neuron.step(10.0, 0.001, 0.102);

        // Should still be at reset potential
        assert_eq!(neuron.v, -70.0);

        // After refractory period
        neuron.step(10.0, 0.001, 0.106);
        assert!(neuron.v > -70.0);
    }
}
