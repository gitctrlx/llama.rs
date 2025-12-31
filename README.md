# llama.rs

A pure Rust implementation of the LLaMA model for inference and educational purposes. Supports LLaMA 1, 2, and 3 architectures.

This repository demonstrates how to run LLaMA inference with minimal dependencies, making it ideal for learning and understanding transformer internals.

## Features

- **HF-aligned Architecture** – Matches **`HuggingFace`** reference implementation with clean, structured codebase matching official model layouts
- **Parallel MHA** – Multi-head attention parallelized with **Rayon** for 2-4x speedup on multi-core systems
- **Minimal Dependencies** – Only uses `byteorder`, `rayon`, `rand`, and `thiserror`
- **Educational** – Line-by-line readable transformer implementation with inline documentation
- **Type-safe** – Leverages Rust's type system for memory safety without garbage collection overhead

## Usage

```sh
cargo run --release -- <checkpoint> <tokenizer> [prompt] [options]
```

### Options

| Flag | Description | Default |
| ------ | ----------- | --------- |
| `--temp <float>` | Sampling temperature (0 = greedy) | 1.0 |
| `--topp <float>` | Top-p (nucleus) sampling | 0.9 |
| `--steps <int>` | Max tokens to generate | 256 |
| `--seed <int>` | Random seed | 0 |

### Example

```sh
cargo run --release -- stories15M.bin tokenizer.bin "Once upon a time" --temp 0.8 --steps 128
```

The examples use small models trained by [`Andrej Karpathy`](https://github.com/karpathy/llama2.c?tab=readme-ov-file#models) for demonstration.

## Related Work

If you're interested in LLaMA implementations in other languages:

- **[llama.go](https://github.com/gitctrlx/llama.go)** – Pure Go implementation
- **[llama.np](https://github.com/gitctrlx/llama.np)** – NumPy-based implementation
- **[llama.cu](https://github.com/gitctrlx/llama.cu)** – CUDA-accelerated implementation

## License

This project is licensed under either of the following licenses, at your option:

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0))
- MIT license ([LICENSE-MIT](LICENSE-MIT) or [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT))

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in `llama.rs` by you, as defined in the Apache-2.0 license, shall be dually licensed as above, without any additional terms or conditions.
