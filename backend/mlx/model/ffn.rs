//! SwiGLU feed-forward network used in Llama-style models.

use mlx_rs::Array;

/// SwiGLU FFN: down(silu(gate(x)) * up(x))
///
/// Weight shapes (out_features, in_features):
/// - gate_proj: (intermediate_size, hidden_size)
/// - up_proj:   (intermediate_size, hidden_size)
/// - down_proj: (hidden_size, intermediate_size)
pub struct FeedForward {
    pub gate_proj: Array,
    pub up_proj: Array,
    pub down_proj: Array,
}

impl FeedForward {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        let gate_proj =
            Array::zeros::<f32>(&[intermediate_size as i32, hidden_size as i32]).unwrap();
        let up_proj = Array::zeros::<f32>(&[intermediate_size as i32, hidden_size as i32]).unwrap();
        let down_proj =
            Array::zeros::<f32>(&[hidden_size as i32, intermediate_size as i32]).unwrap();
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// `x`: (batch, seq_len, hidden_size) → same shape out.
    pub fn forward(&self, x: &Array) -> Array {
        let gate_t = self.gate_proj.transpose_axes(&[1, 0]).unwrap();
        let up_t = self.up_proj.transpose_axes(&[1, 0]).unwrap();
        let down_t = self.down_proj.transpose_axes(&[1, 0]).unwrap();

        // gate = silu(x @ gate_proj^T)
        let gate = x.matmul(&gate_t).unwrap();
        let gate = silu(&gate);

        // up = x @ up_proj^T
        let up = x.matmul(&up_t).unwrap();

        // (gate * up) @ down_proj^T
        let hidden = gate.multiply(&up).unwrap();
        hidden.matmul(&down_t).unwrap()
    }
}

/// SiLU activation: x * sigmoid(x)
fn silu(x: &Array) -> Array {
    let sig = mlx_rs::ops::sigmoid(x).unwrap();
    x.multiply(&sig).unwrap()
}
