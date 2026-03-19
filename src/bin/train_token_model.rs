//! Train a token-level Markov model from JSONL or CSV (e.g. command column).

use anyhow::Result;
use anomark::{
    load_csv, load_jsonl, maybe_filter_training_lines, ModelHandler, TokenMarkovModel, Tokenizer,
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "train-token-model")]
#[command(about = "Train a token-level Markov model from JSONL or CSV", long_about = None)]
struct Args {
    #[arg(short, long)]
    data: String,

    #[arg(short, long, default_value = "command")]
    column: String,

    #[arg(short, long)]
    order: usize,

    /// Tokenizer: whitespace, path, or whitespace_and_path
    #[arg(long, default_value = "whitespace")]
    tokenizer: String,

    #[arg(long)]
    output: Option<String>,

    /// Input format: csv, jsonl, or auto
    #[arg(long, default_value = "auto")]
    format: String,

    /// Exclude Linux kernel-style thread names: entire command is `[name]`
    #[arg(long)]
    exclude_kernel_threads: bool,

    /// Exclude commands matching this regex (repeatable)
    #[arg(long = "exclude-regex", value_name = "PATTERN")]
    exclude_regex: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let is_jsonl = match args.format.to_lowercase().as_str() {
        "jsonl" => true,
        "csv" => false,
        _ => args.data.ends_with(".jsonl"),
    };

    let tokenizer = match args.tokenizer.to_lowercase().as_str() {
        "path" => Tokenizer::PathSegments,
        "whitespace_and_path" | "path_and_whitespace" => Tokenizer::WhitespaceAndPath,
        _ => Tokenizer::Whitespace,
    };

    println!("Loading data from {} ({})...", args.data, if is_jsonl { "JSONL" } else { "CSV" });
    let commands: Vec<String> = if is_jsonl {
        load_jsonl(&args.data, &args.column)?
    } else {
        load_csv(&args.data, &args.column)?
    };
    println!("Loaded {} records, training token model (order={})...", commands.len(), args.order);

    let (commands, _, excluded_train) = maybe_filter_training_lines(
        commands,
        None,
        args.exclude_kernel_threads,
        &args.exclude_regex,
    )?;
    if excluded_train > 0 {
        println!(
            "Excluded {} command(s) from training (--exclude-kernel-threads / --exclude-regex)",
            excluded_train
        );
    }
    if commands.is_empty() {
        anyhow::bail!("No training lines left after exclusions; relax filters or check input.");
    }

    let mut model = TokenMarkovModel::new(args.order, tokenizer);
    for cmd in &commands {
        model.train(cmd, 1);
    }
    model.normalize_model_and_compute_prior();

    ModelHandler::save_token_model(&model, args.output.as_deref())?;
    Ok(())
}
