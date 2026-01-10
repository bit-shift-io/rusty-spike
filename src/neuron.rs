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
#[derive(Clone)]
pub struct LIFNeuron {
    /// Current membrane potential (V)
    pub v: f64,
    /// Resting potential (V)
    pub v_rest: f64,
    /// Reset potential after a spike (V)
    pub v_reset: f64,
    /// Threshold potential to trigger a spike (V)
    pub v_threshold: f64,
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
        v_threshold: f64,
        tau_m: f64,
        r: f64,
        refractory_period: f64,
    ) -> Self {
        Self {
            v: v_rest,
            v_rest,
            v_reset,
            v_threshold,
            tau_m,
            r,
            refractory_period,
            last_spike_time: -f64::INFINITY,
        }
    }

    pub fn default() -> Self {
        Self::new(
            -70.0, // v_rest
            -70.0, // v_reset
            -50.0, // v_threshold
            0.02,  // tau_m
            10.0,  // r
            0.005, // refractory_period
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
        // Check if we are in the refractory period
        if current_time < self.last_spike_time + self.refractory_period {
            self.v = self.v_reset;
            return false;
        }

        // Numerical integration using Euler method:
        // dv = [-(v - v_rest) + R * I] * (dt / tau_m)
        let dv = (-(self.v - self.v_rest) + self.r * i_ext) * (dt / self.tau_m);
        self.v += dv;

        // Check for spike
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.last_spike_time = current_time;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passive_leak() {
        let mut neuron = LIFNeuron::new(-70.0, -70.0, -50.0, 0.02, 10.0, 0.005);
        neuron.v = -60.0; // Start above rest

        // Step forward without input
        neuron.step(0.0, 0.001, 0.001);

        // Potential should have decreased towards -70.0
        assert!(neuron.v < -60.0);
        assert!(neuron.v > -70.0);
    }

    #[test]
    fn test_integration() {
        let mut neuron = LIFNeuron::new(-70.0, -70.0, -50.0, 0.02, 10.0, 0.005);

        // Apply positive current
        neuron.step(2.0, 0.001, 0.001);

        // Potential should have increased
        assert!(neuron.v > -70.0);
    }

    #[test]
    fn test_firing() {
        let mut neuron = LIFNeuron::new(-70.0, -70.0, -50.0, 0.02, 10.0, 0.005);

        // Apply very large current to trigger immediate spike
        let fired = neuron.step(10.0, 0.1, 0.1);

        assert!(fired);
        assert_eq!(neuron.v, -70.0); // Should be reset
        assert_eq!(neuron.last_spike_time, 0.1);
    }

    #[test]
    fn test_refractory_period() {
        let mut neuron = LIFNeuron::new(-70.0, -70.0, -50.0, 0.02, 10.0, 0.005);

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
