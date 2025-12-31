//! Llama model configuration.

/// Transformer hyperparameters, aligned with LlamaConfig in Hugging Face Transformers.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LlamaConfig {
    /// Transformer embedding dimension (hidden_size)
    pub dim: i32,
    /// FFN intermediate dimension (intermediate_size)
    pub hidden_dim: i32,
    /// Number of decoder layers (num_hidden_layers)
    pub n_layers: i32,
    /// Number of query attention heads (num_attention_heads)
    pub n_heads: i32,
    /// Number of key/value heads for GQA (num_key_value_heads)
    pub n_kv_heads: i32,
    /// Vocabulary size (vocab_size)
    pub vocab_size: i32,
    /// Maximum context length (max_position_embeddings)
    pub seq_len: i32,
}

impl LlamaConfig {
    /// Returns the key/value dimension per head group.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        ((self.dim * self.n_kv_heads) / self.n_heads) as usize
    }

    /// Returns the head size.
    #[inline]
    pub fn head_size(&self) -> usize {
        (self.dim / self.n_heads) as usize
    }

    /// Returns the number of heads per KV group (for GQA).
    #[inline]
    pub fn group_size(&self) -> usize {
        (self.n_heads / self.n_kv_heads) as usize
    }
}
