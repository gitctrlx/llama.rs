use llama_rs::{LlamaState, forward, load_model, load_tokenizer, sample};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::env;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <checkpoint> <tokenizer> [prompt] [options]",
            args[0]
        );
        eprintln!("Options:");
        eprintln!("  --temp <float>    Temperature (default: 1.0, 0 = greedy)");
        eprintln!("  --topp <float>    Top-p sampling (default: 0.9)");
        eprintln!("  --steps <int>     Max tokens to generate (default: 256)");
        eprintln!("  --seed <int>      Random seed (default: 0)");
        std::process::exit(1);
    }

    let checkpoint_path = &args[1];
    let tokenizer_path = &args[2];
    let prompt = args.get(3).map(|s| s.as_str()).unwrap_or("");

    // Parse optional arguments
    let mut temp = 1.0;
    let mut topp = 0.9;
    let mut steps = 256usize;
    let mut seed = 0u64;

    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--temp" => {
                temp = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
                i += 2;
            }
            "--topp" => {
                topp = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0.9);
                i += 2;
            }
            "--steps" => {
                steps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(256);
                i += 2;
            }
            "--seed" => {
                seed = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            _ => i += 1,
        }
    }

    // Load model and tokenizer
    eprintln!("Loading model from: {}", checkpoint_path);
    let (config, weights) = load_model(checkpoint_path)?;
    eprintln!(
        "Config: dim={}, layers={}, heads={}, vocab={}",
        config.dim, config.n_layers, config.n_heads, config.vocab_size
    );

    let tokenizer = load_tokenizer(tokenizer_path, config.vocab_size as usize)?;
    eprintln!("Loaded tokenizer with {} tokens", tokenizer.vocab.len());

    // Initialize state and RNG
    let mut state = LlamaState::new(&config);
    let mut rng = StdRng::seed_from_u64(seed);

    // Encode prompt
    let tokens = tokenizer.encode(prompt, true, false)?;
    eprintln!("Prompt tokens: {:?}", tokens);

    // Generate
    let mut pos = 0i32;
    let mut token = tokens[0];

    for step in 0..steps {
        forward(token, pos, &config, &mut state, &weights);

        let next_token = if step < tokens.len() - 1 {
            tokens[step + 1]
        } else {
            sample(&mut state.logits, temp, topp, &mut rng)
        };

        // Decode and print token
        if let Some(piece) = tokenizer.decode(next_token) {
            // Handle special byte tokens (encoded as <0xXX>)
            if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
                if let Ok(byte) = u8::from_str_radix(&piece[3..5], 16) {
                    print!("{}", byte as char);
                }
            } else {
                print!("{}", piece);
            }
            io::stdout().flush()?;
        }

        // Check for EOS
        if next_token == 2 {
            break;
        }

        token = next_token;
        pos += 1;
    }

    println!();
    Ok(())
}
