/// Global optical-flow approximation using average motion vectors between predicted
/// and observed tracks. This does not require image pixels – it measures displacement
/// of detection centers frame-to-frame and smooths it for future predictions.
#[derive(Clone, Debug)]
pub struct OpticalFlowEstimator {
    flow: (f32, f32),
    alpha: f32,
    max_magnitude: f32,
}

impl OpticalFlowEstimator {
    pub fn new(alpha: f32, max_magnitude: f32) -> Self {
        Self {
            flow: (0.0, 0.0),
            alpha: alpha.clamp(0.0, 1.0),
            max_magnitude,
        }
    }

    pub fn update(&mut self, samples: &[(f32, f32)]) {
        if samples.is_empty() {
            return;
        }
        let (sum_x, sum_y) = samples.iter().fold((0.0, 0.0), |acc, (dx, dy)| {
            (acc.0 + dx, acc.1 + dy)
        });
        let count = samples.len() as f32;
        let avg = (sum_x / count, sum_y / count);
        let blended = (
            self.flow.0 * (1.0 - self.alpha) + avg.0 * self.alpha,
            self.flow.1 * (1.0 - self.alpha) + avg.1 * self.alpha,
        );
        self.flow = clamp_vector(blended, self.max_magnitude);
    }

    pub fn current_flow(&self) -> (f32, f32) {
        self.flow
    }
}

fn clamp_vector(v: (f32, f32), max_len: f32) -> (f32, f32) {
    if max_len <= 0.0 {
        return v;
    }
    let len = (v.0 * v.0 + v.1 * v.1).sqrt();
    if len <= max_len || len <= f32::EPSILON {
        v
    } else {
        let scale = max_len / len;
        (v.0 * scale, v.1 * scale)
    }
}
