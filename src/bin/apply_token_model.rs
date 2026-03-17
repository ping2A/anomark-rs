//! Apply a trained token-level Markov model to detect anomalies (JSONL or CSV).

use anyhow::Result;
use anomark::{
    load_csv_with_columns, load_jsonl_with_columns,
    ModelHandler,
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "apply-token-model")]
#[command(about = "Apply a token-level Markov model to detect anomalies", long_about = None)]
struct Args {
    #[arg(short, long)]
    data: String,

    #[arg(long, default_value = "auto")]
    format: String,

    #[arg(short, long)]
    model: String,

    #[arg(short, long, default_value = "command")]
    column: String,

    #[arg(short, long)]
    store: bool,

    #[arg(short, long)]
    output: Option<String>,

    #[arg(short = 'n', long, default_value = "50")]
    n_lines: usize,

    #[arg(long)]
    silent: bool,

    #[arg(long)]
    explain: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.store && args.output.is_some() {
        anyhow::bail!("--store and --output cannot be used together");
    }

    println!("Loading token model from {}...", args.model);
    let model = ModelHandler::load_token_model(&args.model)?;

    let is_jsonl = match args.format.to_lowercase().as_str() {
        "jsonl" => true,
        "csv" => false,
        _ => args.data.ends_with(".jsonl"),
    };
    println!("Loading data from {}...", args.data);
    let data = if is_jsonl {
        load_jsonl_with_columns(&args.data)?
    } else {
        load_csv_with_columns(&args.data)?
    };

    let results = ModelHandler::execute_on_data_token(
        &model,
        data,
        &args.column,
        false,
        false,
        args.explain,
    )?;

    if !args.silent {
        ModelHandler::display_top(
            &results,
            &anomark::MarkovModel::new(1), // dummy for threshold display; we don't use color/percentage
            args.n_lines,
            false,
            false,
            args.explain,
        );
    }

    if args.store || args.output.is_some() {
        ModelHandler::save_results(
            &results,
            args.output.as_deref(),
            &anomark::MarkovModel::new(1),
            false,
            false,
            args.explain,
        )?;
    }

    Ok(())
}
