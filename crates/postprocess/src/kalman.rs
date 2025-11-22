use std::f32::EPSILON;

/// Lightweight constant-velocity Kalman filter for 1D motion (position + velocity).
/// Used twice (x/y) to smooth bounding box centers.
#[derive(Clone, Debug)]
pub struct Kalman1D {
    pub state: [f32; 2],      // [position, velocity]
    pub covariance: [[f32; 2]; 2],
    process_noise: f32,
    measurement_noise: f32,
}

impl Kalman1D {
    pub fn new(position: f32, velocity: f32, process_noise: f32, measurement_noise: f32) -> Self {
        Self {
            state: [position, velocity],
            covariance: [[1.0, 0.0], [0.0, 1.0]],
            process_noise,
            measurement_noise: measurement_noise.max(EPSILON),
        }
    }

    /// Predict next state with sampling period dt (frames assumed at 1.0f if unknown)
    pub fn predict(&mut self, dt: f32) {
        let dt = if dt <= 0.0 { 1.0 } else { dt };
        let pos = self.state[0] + self.state[1] * dt;
        self.state[0] = pos;

        // Covariance prediction: P = F P F^T + Q, where F = [[1, dt], [0, 1]]
        let p00 = self.covariance[0][0];
        let p01 = self.covariance[0][1];
        let p10 = self.covariance[1][0];
        let p11 = self.covariance[1][1];

        self.covariance[0][0] = p00 + dt * (p10 + p01) + dt * dt * p11 + self.process_noise;
        self.covariance[0][1] = p01 + dt * p11;
        self.covariance[1][0] = p10 + dt * p11;
        self.covariance[1][1] = p11 + self.process_noise * 0.25;
    }

    /// Correct with measured position.
    pub fn correct(&mut self, measurement: f32) {
        // Innovation
        let innovation = measurement - self.state[0];

        // Innovation covariance S = HPH^T + R (H = [1 0])
        let s = self.covariance[0][0] + self.measurement_noise;
        if s.abs() < EPSILON {
            return;
        }

        // Kalman gain K = P H^T S^-1 -> [p00 / s, p10 / s]^T
        let k0 = self.covariance[0][0] / s;
        let k1 = self.covariance[1][0] / s;

        // Update state
        self.state[0] += k0 * innovation;
        self.state[1] += k1 * innovation;

        // Update covariance: P = (I - K H) P
        let p00 = self.covariance[0][0];
        let p01 = self.covariance[0][1];
        let p10 = self.covariance[1][0];
        let p11 = self.covariance[1][1];

        self.covariance[0][0] = (1.0 - k0) * p00;
        self.covariance[0][1] = (1.0 - k0) * p01;
        self.covariance[1][0] = p10 - k1 * p00;
        self.covariance[1][1] = p11 - k1 * p01;
    }

    pub fn position(&self) -> f32 {
        self.state[0]
    }
}

#[derive(Clone, Debug)]
pub struct KalmanFilter2D {
    x_filter: Kalman1D,
    y_filter: Kalman1D,
}

impl KalmanFilter2D {
    pub fn new(cx: f32, cy: f32, process_noise: f32, measurement_noise: f32) -> Self {
        Self {
            x_filter: Kalman1D::new(cx, 0.0, process_noise, measurement_noise),
            y_filter: Kalman1D::new(cy, 0.0, process_noise, measurement_noise),
        }
    }

    pub fn predict(&mut self, dt: f32) -> (f32, f32) {
        self.x_filter.predict(dt);
        self.y_filter.predict(dt);
        (self.x_filter.position(), self.y_filter.position())
    }

    pub fn correct(&mut self, cx: f32, cy: f32) {
        self.x_filter.correct(cx);
        self.y_filter.correct(cy);
    }

    pub fn position(&self) -> (f32, f32) {
        (self.x_filter.position(), self.y_filter.position())
    }
}
