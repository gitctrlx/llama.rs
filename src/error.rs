//! Error types for Llama inference.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlamaError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid model file: {0}")]
    InvalidModel(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
}

pub type Result<T> = std::result::Result<T, LlamaError>;
