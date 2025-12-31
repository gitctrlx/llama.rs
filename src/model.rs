//! Llama model forward pass.

use crate::config::LlamaConfig;
use crate::error::Result;
use crate::ops::{accum, apply_rotary_emb, matmul, rms_norm, softmax, swiglu};
use crate::state::LlamaState;
use crate::weights::{LlamaLayerWeights, LlamaWeights};
use byteorder::{LittleEndian, ReadBytesExt};
use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Load config and weights from a binary checkpoint file.
pub fn load_model<P: AsRef<Path>>(path: P) -> Result<(LlamaConfig, LlamaWeights)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let config = LlamaConfig {
        dim: reader.read_i32::<LittleEndian>()?,
        hidden_dim: reader.read_i32::<LittleEndian>()?,
        n_layers: reader.read_i32::<LittleEndian>()?,
        n_heads: reader.read_i32::<LittleEndian>()?,
        n_kv_heads: reader.read_i32::<LittleEndian>()?,
        vocab_size: reader.read_i32::<LittleEndian>()?,
        seq_len: reader.read_i32::<LittleEndian>()?,
    };

    let weights = LlamaWeights::load(&mut reader, &config)?;

    Ok((config, weights))
}

/// Perform a single-token forward pass, aligned with LlamaModel.forward.
pub fn forward(
    token: i32,
    pos: i32,
    config: &LlamaConfig,
    state: &mut LlamaState,
    weights: &LlamaWeights,
) {
    let dim = config.dim as usize;

    // Token embedding
    let emb_offset = (token as usize) * dim;
    state
        .x
        .copy_from_slice(&weights.embed_tokens[emb_offset..emb_offset + dim]);

    // Decoder layers
    for l in 0..config.n_layers as usize {
        attention(l, pos, config, state, &weights.layers[l]);
        mlp(config, state, &weights.layers[l]);
    }

    // Final norm
    let x_clone = state.x.clone();
    rms_norm(&mut state.x, &x_clone, &weights.norm);

    // Logits (using tied embeddings)
    matmul(&mut state.logits, &state.x, &weights.embed_tokens);
}

/// Self-attention for one layer, aligned with LlamaAttention.forward.
fn attention(
    layer_idx: usize,
    pos: i32,
    config: &LlamaConfig,
    state: &mut LlamaState,
    layer_weights: &LlamaLayerWeights,
) {
    let _dim = config.dim as usize;
    let n_heads = config.n_heads as usize;
    let head_size = config.head_size();
    let kv_dim = config.kv_dim();
    let group_size = config.group_size();

    // Input norm
    rms_norm(&mut state.xb, &state.x, &layer_weights.attn_norm);

    // QKV projections
    matmul(&mut state.q, &state.xb, &layer_weights.q_proj);
    matmul(&mut state.k, &state.xb, &layer_weights.k_proj);
    matmul(&mut state.v, &state.xb, &layer_weights.v_proj);

    // Apply RoPE
    apply_rotary_emb(&mut state.q, pos, head_size);
    apply_rotary_emb(&mut state.k, pos, head_size);

    // Cache K and V
    let cache_offset = (pos as usize) * kv_dim;
    state.key_cache[layer_idx][cache_offset..cache_offset + kv_dim].copy_from_slice(&state.k);
    state.value_cache[layer_idx][cache_offset..cache_offset + kv_dim].copy_from_slice(&state.v);

    // Multi-head attention (parallelized)
    let key_cache = &state.key_cache[layer_idx];
    let value_cache = &state.value_cache[layer_idx];

    // Collect results from parallel computation
    let head_outputs: Vec<Vec<f32>> = (0..n_heads)
        .into_par_iter()
        .map(|h| {
            let q_off = h * head_size;
            let q = &state.q[q_off..q_off + head_size];
            let kv_h = h / group_size;

            // Compute attention scores
            let mut att = vec![0.0f32; (pos + 1) as usize];
            for t in 0..=pos as usize {
                let k_off = t * kv_dim + kv_h * head_size;
                let k = &key_cache[k_off..k_off + head_size];

                let mut score = 0.0f32;
                for i in 0..head_size {
                    score += q[i] * k[i];
                }
                att[t] = score / (head_size as f32).sqrt();
            }

            // Softmax
            softmax(&mut att);

            // Weighted sum of values
            let mut out = vec![0.0f32; head_size];
            for t in 0..=pos as usize {
                let v_off = t * kv_dim + kv_h * head_size;
                let v = &value_cache[v_off..v_off + head_size];
                let a = att[t];
                for i in 0..head_size {
                    out[i] += a * v[i];
                }
            }
            out
        })
        .collect();

    // Gather results into xb
    for (h, out) in head_outputs.into_iter().enumerate() {
        let xb_off = h * head_size;
        state.xb[xb_off..xb_off + head_size].copy_from_slice(&out);
    }

    // Output projection
    matmul(&mut state.xb2, &state.xb, &layer_weights.o_proj);

    // Residual add
    accum(&mut state.x, &state.xb2);
}

/// FFN for one layer, aligned with LlamaMLP.forward.
fn mlp(_config: &LlamaConfig, state: &mut LlamaState, layer_weights: &LlamaLayerWeights) {
    // Input norm
    rms_norm(&mut state.xb, &state.x, &layer_weights.ffn_norm);

    // Gate and up projections
    matmul(&mut state.hb, &state.xb, &layer_weights.gate_proj);
    matmul(&mut state.hb2, &state.xb, &layer_weights.up_proj);

    // SwiGLU activation
    swiglu(&mut state.hb, &state.hb2);

    // Down projection
    matmul(&mut state.xb, &state.hb, &layer_weights.down_proj);

    // Residual add
    accum(&mut state.x, &state.xb);
}
