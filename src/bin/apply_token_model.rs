//! Apply a trained token-level Markov model to detect anomalies (JSONL or CSV).

use anyhow::Result;
use anomark::{
    load_csv_with_columns, load_jsonl_with_columns, ModelHandler, TrainLineFilter,
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

    /// Skip rows whose command is a kernel-thread-style name (e.g. [kthreadd])
    #[arg(long)]
    exclude_kernel_threads: bool,

    #[arg(long = "exclude-regex", value_name = "PATTERN")]
    exclude_regex: Vec<String>,

    /// Column/field that contains the machine/host name; output will include a Machine column
    #[arg(long = "machine-field", value_name = "COLUMN")]
    machine_field: Option<String>,

    /// Use this value as Machine for every row
    #[arg(long, value_name = "NAME")]
    machine: Option<String>,
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

    let exclude_filter = if args.exclude_kernel_threads || !args.exclude_regex.is_empty() {
        Some(TrainLineFilter::new(args.exclude_kernel_threads, &args.exclude_regex)?)
    } else {
        None
    };

    let results = ModelHandler::execute_on_data_token(
        &model,
        data,
        &args.column,
        false,
        false,
        args.explain,
        exclude_filter.as_ref(),
        args.machine_field.as_deref(),
        args.machine.as_deref(),
    )?;

    let suspect_ln = model.prior.ln() * 0.95;

    if !args.silent {
        ModelHandler::display_top(
            &results,
            &anomark::MarkovModel::new(1), // dummy for color/percentage (unused here)
            args.n_lines,
            false,
            false,
            args.explain,
            Some(suspect_ln),
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
            Some(suspect_ln),
        )?;
    }

    Ok(())
}
