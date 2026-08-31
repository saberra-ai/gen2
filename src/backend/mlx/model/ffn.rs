//! SwiGLU feed-forward network used in Llama-style models.

use mlx_rs::Array;

use super::quantized::Weight;

/// SwiGLU FFN: down(silu(gate(x)) * up(x))
///
/// Weight shapes (out_features, in_features):
/// - gate_proj: (intermediate_size, hidden_size)
/// - up_proj:   (intermediate_size, hidden_size)
/// - down_proj: (hidden_size, intermediate_size)
pub struct FeedForward {
    pub gate_proj: Weight,
    pub up_proj: Weight,
    pub down_proj: Weight,
}

impl FeedForward {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        let gate_proj = Weight::plain(
            Array::zeros::<f32>(&[intermediate_size as i32, hidden_size as i32]).expect("mlx op"),
        );
        let up_proj = Weight::plain(
            Array::zeros::<f32>(&[intermediate_size as i32, hidden_size as i32]).expect("mlx op"),
        );
        let down_proj = Weight::plain(
            Array::zeros::<f32>(&[hidden_size as i32, intermediate_size as i32]).expect("mlx op"),
        );
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// `x`: (batch, seq_len, hidden_size) → same shape out.
    pub fn forward(&self, x: &Array) -> Array {
        // gate = silu(x @ gate_proj^T)
        let gate = self.gate_proj.matmul_transpose(x);
        let gate = silu(&gate);

        // up = x @ up_proj^T
        let up = self.up_proj.matmul_transpose(x);

        // (gate * up) @ down_proj^T
        let hidden = gate.multiply(&up).expect("mlx op");
        self.down_proj.matmul_transpose(&hidden)
    }
}

/// SiLU activation: x * sigmoid(x)
fn silu(x: &Array) -> Array {
    let sig = mlx_rs::ops::sigmoid(x).expect("mlx op");
    x.multiply(&sig).expect("mlx op")
}
