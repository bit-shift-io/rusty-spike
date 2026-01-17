use std::f64;

/// A trait for encoding continuous values into spikes.
pub trait Encoder {
    /// Takes an input value and returns whether a spike occurred at the current time step.
    fn step(&mut self, input: f64, dt: f64, current_time: f64) -> bool;
}

/// Rate Coding: Converts input intensity into a spike train.
/// Uses a deterministic accumulator-based approach for more stability.
#[derive(Clone)]
pub struct RateEncoder {
    /// Gain factor for the input (maps input to firing rate in Hz)
    pub gain: f64,
    /// Accumulator for deterministic firing
    pub accumulator: f64,
}

impl RateEncoder {
    pub fn new(gain: f64) -> Self {
        Self {
            gain,
            accumulator: 0.0,
        }
    }

    /// Resets the internal accumulator state.
    pub fn reset(&mut self) {
        self.accumulator = 0.0;
    }
}

impl Encoder for RateEncoder {
    fn step(&mut self, input: f64, dt: f64, _current_time: f64) -> bool {
        // Increment accumulator based on rate and dt
        // rate = gain * input
        // amount_to_add = rate * dt
        self.accumulator += self.gain * input * dt;

        if self.accumulator >= 1.0 {
            self.accumulator -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Latency Coding: Input intensity determines the timing of the first spike.
/// Higher input results in an earlier spike.
#[allow(dead_code)]
#[derive(Clone)]
pub struct LatencyEncoder {
    /// Time constant for the latency decay
    pub tau: f64,
    /// Has the spike already occurred?
    pub spiked: bool,
    /// Threshold for triggering the spike
    pub threshold: f64,
}

impl LatencyEncoder {
    #[allow(dead_code)]
    pub fn new(tau: f64, threshold: f64) -> Self {
        Self {
            tau,
            spiked: false,
            threshold,
        }
    }

    /// Resets the encoder for a new encoding cycle (e.g., new stimulus).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.spiked = false;
    }
}

impl Encoder for LatencyEncoder {
    fn step(&mut self, input: f64, _dt: f64, current_time: f64) -> bool {
        if self.spiked || input <= 0.0 {
            return false;
        }

        // Latency model: t_spike = tau * ln(input / threshold)
        // For simplicity, we can use a direct mapping: t_spike = 1.0 / (input * gain)
        // Or integrate until a target time is reached.
        // Let's use a simple heuristic based on current_time vs calculated latency.
        let latency = self.tau / input;

        if current_time >= latency {
            self.spiked = true;
            true
        } else {
            false
        }
    }
}

/// Delta Modulation: Emits a spike when the change in input exceeds a threshold.
/// This captures the derivative of the signal.
#[allow(dead_code)]
#[derive(Clone)]
pub struct DeltaEncoder {
    /// Threshold for the change in signal
    pub threshold: f64,
    /// Previous input value
    pub last_input: f64,
}

impl DeltaEncoder {
    #[allow(dead_code)]
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            last_input: 0.0,
        }
    }
}

impl Encoder for DeltaEncoder {
    fn step(&mut self, input: f64, _dt: f64, _current_time: f64) -> bool {
        let delta = (input - self.last_input).abs();
        if delta >= self.threshold {
            self.last_input = input;
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
    fn test_delta_encoder() {
        let mut encoder = DeltaEncoder::new(0.5);

        // Small change, no spike
        assert!(!encoder.step(0.1, 0.001, 0.001));

        // Large change, spike
        assert!(encoder.step(0.7, 0.001, 0.002));

        // Small change relative to new baseline, no spike
        assert!(!encoder.step(0.8, 0.001, 0.003));
    }

    #[test]
    fn test_latency_encoder() {
        let mut encoder = LatencyEncoder::new(1.0, 0.1);

        // Input is 1.0, latency = 1.0 / 1.0 = 1.0
        assert!(!encoder.step(1.0, 0.1, 0.5));
        assert!(encoder.step(1.0, 0.1, 1.1));

        // Should only spike once
        assert!(!encoder.step(1.0, 0.1, 1.2));
    }
}
