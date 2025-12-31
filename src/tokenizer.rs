//! Tokenizer loading and BPE encoding.

use crate::error::{LlamaError, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Tokenizer holding vocabulary and scores.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub scores: Vec<f32>,
    pub vocab_map: HashMap<String, i32>,
    pub max_token_len: u32,
}

impl Tokenizer {
    /// Encode text using BPE, with optional BOS/EOS tokens.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<i32>> {
        bpe_encode(text, &self.vocab, &self.scores, &self.vocab_map, bos, eos)
    }

    /// Decode a token ID to its string representation.
    pub fn decode(&self, token: i32) -> Option<&str> {
        self.vocab.get(token as usize).map(|s| s.as_str())
    }
}

/// Load tokenizer from a binary file.
pub fn load_tokenizer<P: AsRef<Path>>(path: P, vocab_size: usize) -> Result<Tokenizer> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let max_token_len = reader.read_u32::<LittleEndian>()?;

    let mut vocab = Vec::with_capacity(vocab_size);
    let mut scores = Vec::with_capacity(vocab_size);
    let mut vocab_map = HashMap::with_capacity(vocab_size);

    for i in 0..vocab_size {
        let score = reader.read_f32::<LittleEndian>()?;
        scores.push(score);

        let len = reader.read_i32::<LittleEndian>()? as usize;
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;

        let token = String::from_utf8_lossy(&buf).into_owned();
        vocab_map.insert(token.clone(), i as i32);
        vocab.push(token);
    }

    Ok(Tokenizer {
        vocab,
        scores,
        vocab_map,
        max_token_len,
    })
}

/// BPE encode text, aligned with the C implementation.
pub fn bpe_encode(
    text: &str,
    vocab: &[String],
    scores: &[f32],
    vocab_map: &HashMap<String, i32>,
    bos: bool,
    eos: bool,
) -> Result<Vec<i32>> {
    let mut tokens: Vec<i32> = Vec::with_capacity(text.len() + 3);

    // Add BOS token if requested
    if bos {
        tokens.push(1);
    }

    // Add dummy prefix space if text is not empty (llama tokenizer behavior)
    if !text.is_empty() {
        let dummy_prefix = vocab_map.get(" ").ok_or_else(|| {
            LlamaError::Tokenizer("dummy prefix ' ' not found in vocabulary".into())
        })?;
        tokens.push(*dummy_prefix);
    }

    // Process text character by character
    for c in text.chars() {
        let char_str = c.to_string();
        if let Some(&id) = vocab_map.get(&char_str) {
            tokens.push(id);
        } else {
            // Byte-level fallback for unknown characters
            for b in char_str.as_bytes() {
                tokens.push(*b as i32 + 3);
            }
        }
    }

    // Iteratively merge the best pair
    loop {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_id = -1i32;
        let mut best_idx = None;

        for i in 0..tokens.len().saturating_sub(1) {
            let merged = format!(
                "{}{}",
                vocab[tokens[i] as usize],
                vocab[tokens[i + 1] as usize]
            );
            if let Some(&id) = vocab_map.get(&merged) {
                if scores[id as usize] > best_score {
                    best_score = scores[id as usize];
                    best_id = id;
                    best_idx = Some(i);
                }
            }
        }

        let Some(idx) = best_idx else { break };

        // Merge the best pair
        tokens[idx] = best_id;
        tokens.remove(idx + 1);
    }

    // Add EOS token if requested
    if eos {
        tokens.push(2);
    }

    Ok(tokens)
}
