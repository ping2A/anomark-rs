//! Print summary information about a saved character or token model.

use anyhow::{Context, Result};
use anomark::{ModelHandler, Tokenizer};
use clap::Parser;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "inspect-model")]
#[command(about = "Display metadata and statistics for a trained model file", long_about = None)]
struct Args {
    /// Path to a .bin model (character or token)
    #[arg(short, long)]
    model: PathBuf,

    /// Emit machine-readable JSON
    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let path = args.model.to_str().context("Invalid path")?;
    let meta = fs::metadata(path).with_context(|| format!("Cannot read {}", path))?;
    let bytes = meta.len();

    let char_result = ModelHandler::load_model(path);
    let token_result = ModelHandler::load_token_model(path);

    if args.json {
        if let Ok(m) = char_result {
            let s = serde_json::json!({
                "kind": "character",
                "path": path,
                "file_bytes": bytes,
                "order": m.order,
                "prior": m.prior,
                "trained": m.is_trained(),
                "num_contexts": m.num_contexts(),
                "num_transitions": m.num_transitions(),
                "alphabet_len": m.alphabet_len(),
            });
            println!("{}", serde_json::to_string_pretty(&s)?);
            return Ok(());
        }
        if let Ok(t) = token_result {
            let tokenizer = match &t.tokenizer {
                Tokenizer::Whitespace => "whitespace",
                Tokenizer::PathSegments => "path",
                Tokenizer::WhitespaceAndPath => "whitespace_and_path",
            };
            let s = serde_json::json!({
                "kind": "token",
                "path": path,
                "file_bytes": bytes,
                "order": t.order,
                "prior": t.prior,
                "trained": t.is_trained(),
                "tokenizer": tokenizer,
                "num_contexts": t.num_contexts(),
                "num_transitions": t.num_transitions(),
            });
            println!("{}", serde_json::to_string_pretty(&s)?);
            return Ok(());
        }
        anyhow::bail!("Could not load as character or token model (wrong format or corrupt file)");
    }

    println!("Model file: {}", path);
    println!("File size:  {} bytes", bytes);

    if let Ok(m) = char_result {
        println!();
        println!("Type:       Character-level Markov");
        println!("Order:      {} (context length in characters)", m.order);
        println!("Prior:      {:.6e}", m.prior);
        println!("Trained:    {}", if m.is_trained() { "yes" } else { "no (empty chain)" });
        println!("Contexts:   {} distinct n-gram prefixes", m.num_contexts());
        println!("Transitions: {} (context → next-char edges)", m.num_transitions());
        println!("Alphabet:   {} distinct characters", m.alphabet_len());
        return Ok(());
    }

    if let Ok(t) = token_result {
        let tokenizer = match &t.tokenizer {
            Tokenizer::Whitespace => "whitespace",
            Tokenizer::PathSegments => "path",
            Tokenizer::WhitespaceAndPath => "whitespace_and_path",
        };
        println!();
        println!("Type:       Token-level Markov");
        println!("Order:      {} (context length in tokens)", t.order);
        println!("Tokenizer:  {}", tokenizer);
        println!("Prior:      {:.6e}", t.prior);
        println!("Trained:    {}", if t.is_trained() { "yes" } else { "no (empty chain)" });
        println!("Contexts:   {} distinct token contexts", t.num_contexts());
        println!("Transitions: {} (context → next-token edges)", t.num_transitions());
        return Ok(());
    }

    anyhow::bail!(
        "Could not load as character or token model.\n\
         Hint: models from `train` (character) are distinct from `train-token-model`;\n\
         models from train-token-model are token models."
    );
}
