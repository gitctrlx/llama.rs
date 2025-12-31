//! Model weights for Llama.

use crate::config::LlamaConfig;
use crate::error::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;

/// Weights for a single decoder layer.
#[derive(Debug, Clone)]
pub struct LlamaLayerWeights {
    /// Input RMSNorm weights (input_layernorm)
    pub attn_norm: Vec<f32>,
    /// Query projection (self_attn.q_proj.weight)
    pub q_proj: Vec<f32>,
    /// Key projection (self_attn.k_proj.weight)
    pub k_proj: Vec<f32>,
    /// Value projection (self_attn.v_proj.weight)
    pub v_proj: Vec<f32>,
    /// Output projection (self_attn.o_proj.weight)
    pub o_proj: Vec<f32>,
    /// Post-attention RMSNorm weights (post_attention_layernorm)
    pub ffn_norm: Vec<f32>,
    /// Gate projection in MLP (mlp.gate_proj.weight)
    pub gate_proj: Vec<f32>,
    /// Up projection in MLP (mlp.up_proj.weight)
    pub up_proj: Vec<f32>,
    /// Down projection in MLP (mlp.down_proj.weight)
    pub down_proj: Vec<f32>,
}

/// All model parameters, aligned with LlamaModel weights in Transformers.
#[derive(Debug, Clone)]
pub struct LlamaWeights {
    /// Token embeddings (model.embed_tokens.weight)
    pub embed_tokens: Vec<f32>,
    /// Decoder layers (model.layers)
    pub layers: Vec<LlamaLayerWeights>,
    /// Final RMSNorm (model.norm.weight)
    pub norm: Vec<f32>,
}

impl LlamaWeights {
    /// Load weights from a binary reader.
    pub fn load<R: Read>(reader: &mut R, config: &LlamaConfig) -> Result<Self> {
        let dim = config.dim as usize;
        let hdim = config.hidden_dim as usize;
        let n_layers = config.n_layers as usize;
        let vocab = config.vocab_size as usize;
        let kv_dim = config.kv_dim();

        // Read embed_tokens
        let embed_tokens = read_f32_vec(reader, vocab * dim)?;

        // Read flat weight buffers
        let rms_att_flat = read_f32_vec(reader, n_layers * dim)?;
        let wq_flat = read_f32_vec(reader, n_layers * dim * dim)?;
        let wk_flat = read_f32_vec(reader, n_layers * dim * kv_dim)?;
        let wv_flat = read_f32_vec(reader, n_layers * dim * kv_dim)?;
        let wo_flat = read_f32_vec(reader, n_layers * dim * dim)?;
        let rms_ffn_flat = read_f32_vec(reader, n_layers * dim)?;
        let gate_flat = read_f32_vec(reader, n_layers * hdim * dim)?;
        let down_flat = read_f32_vec(reader, n_layers * dim * hdim)?;
        let up_flat = read_f32_vec(reader, n_layers * hdim * dim)?;
        let norm = read_f32_vec(reader, dim)?;

        // Build per-layer weights
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let attn_norm = rms_att_flat[l * dim..(l + 1) * dim].to_vec();
            let q_proj = wq_flat[l * dim * dim..(l + 1) * dim * dim].to_vec();
            let k_proj = wk_flat[l * dim * kv_dim..(l + 1) * dim * kv_dim].to_vec();
            let v_proj = wv_flat[l * dim * kv_dim..(l + 1) * dim * kv_dim].to_vec();
            let o_proj = wo_flat[l * dim * dim..(l + 1) * dim * dim].to_vec();
            let ffn_norm = rms_ffn_flat[l * dim..(l + 1) * dim].to_vec();
            let gate_proj = gate_flat[l * hdim * dim..(l + 1) * hdim * dim].to_vec();
            let down_proj = down_flat[l * dim * hdim..(l + 1) * dim * hdim].to_vec();
            let up_proj = up_flat[l * hdim * dim..(l + 1) * hdim * dim].to_vec();

            layers.push(LlamaLayerWeights {
                attn_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                ffn_norm,
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        Ok(LlamaWeights {
            embed_tokens,
            layers,
            norm,
        })
    }
}

/// Read a vector of f32 values from the reader.
fn read_f32_vec<R: Read>(reader: &mut R, count: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0f32; count];
    for v in buf.iter_mut() {
        *v = reader.read_f32::<LittleEndian>()?;
    }
    Ok(buf)
}
