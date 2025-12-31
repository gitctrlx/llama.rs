//! Runtime state buffers for Llama inference.

use crate::config::LlamaConfig;

/// Runtime buffers for inference, aligned with forward pass states.
#[derive(Debug, Clone)]
pub struct LlamaState {
    /// Current hidden state (hidden_states)
    pub x: Vec<f32>,
    /// Buffer for attention output before projection
    pub xb: Vec<f32>,
    /// Temp buffer for attention projection output
    pub xb2: Vec<f32>,
    /// FFN gate activation buffer
    pub hb: Vec<f32>,
    /// FFN up activation buffer
    pub hb2: Vec<f32>,
    /// Query vector
    pub q: Vec<f32>,
    /// Key vector
    pub k: Vec<f32>,
    /// Value vector
    pub v: Vec<f32>,
    /// Attention scores per head [n_heads][seq_len]
    pub att: Vec<Vec<f32>>,
    /// Output logits
    pub logits: Vec<f32>,
    /// Key cache [n_layers][seq_len * kv_dim]
    pub key_cache: Vec<Vec<f32>>,
    /// Value cache [n_layers][seq_len * kv_dim]
    pub value_cache: Vec<Vec<f32>>,
}

impl LlamaState {
    /// Allocate inference buffers based on config.
    pub fn new(config: &LlamaConfig) -> Self {
        let dim = config.dim as usize;
        let hdim = config.hidden_dim as usize;
        let n_heads = config.n_heads as usize;
        let n_layers = config.n_layers as usize;
        let seq_len = config.seq_len as usize;
        let kv_dim = config.kv_dim();
        let vocab_size = config.vocab_size as usize;

        let att = (0..n_heads).map(|_| vec![0.0f32; seq_len]).collect();
        let key_cache = (0..n_layers)
            .map(|_| vec![0.0f32; seq_len * kv_dim])
            .collect();
        let value_cache = (0..n_layers)
            .map(|_| vec![0.0f32; seq_len * kv_dim])
            .collect();

        LlamaState {
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            xb2: vec![0.0; dim],
            hb: vec![0.0; hdim],
            hb2: vec![0.0; hdim],
            q: vec![0.0; dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att,
            logits: vec![0.0; vocab_size],
            key_cache,
            value_cache,
        }
    }
}
