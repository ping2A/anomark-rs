use anyhow::Result;
use anomark::{load_csv_with_columns, load_jsonl_with_columns, ModelHandler};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "apply-model")]
#[command(about = "Apply a trained Markov model to detect anomalies", long_about = None)]
struct Args {
    /// Path of the data to work on (CSV or JSONL file)
    #[arg(short, long)]
    data: String,

    /// Input format: csv, jsonl, or auto (detect from file extension)
    #[arg(long, default_value = "auto")]
    format: String,

    /// Path to the model to use
    #[arg(short, long)]
    model: String,

    /// The column/field name on which we want to execute the model (e.g. "command" for JSONL)
    #[arg(short, long)]
    column: String,

    /// Store output in csv format in the results folder
    #[arg(short, long)]
    store: bool,

    /// Path to file where output in csv format will be stored
    #[arg(short, long)]
    output: Option<String>,

    /// Color the least likely letters in the output
    #[arg(long)]
    color: bool,

    /// The number of lines you want to display
    #[arg(short = 'n', long, default_value = "50")]
    n_lines: usize,

    /// Silent mode (no terminal output)
    #[arg(long)]
    silent: bool,

    /// Apply GUID, SID, username, and hash replacement by placeholder
    #[arg(long)]
    placeholder: bool,

    /// Apply filepath replacement by placeholder
    #[arg(long)]
    filepath_placeholder: bool,

    /// Show human-readable percentage
    #[arg(long)]
    show_percentage: bool,

    /// Explain anomalies: show unusual n-grams that contributed to low score
    #[arg(long)]
    explain: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.store && args.output.is_some() {
        anyhow::bail!("'--store' and '--output' flags cannot be used at the same time.");
    }

    // Load model
    println!("Loading model from {}...", args.model);
    let mut model = ModelHandler::load_model(&args.model)?;

    // Detect format and load data
    let is_jsonl = match args.format.to_lowercase().as_str() {
        "jsonl" => true,
        "csv" => false,
        _ => args.data.ends_with(".jsonl"),
    };

    println!("Loading data from {} ({})...", args.data, if is_jsonl { "JSONL" } else { "CSV" });
    let data = if is_jsonl {
        load_jsonl_with_columns(&args.data)?
    } else {
        load_csv_with_columns(&args.data)?
    };

    // Execute model on data
    let results = ModelHandler::execute_on_data(
        &mut model,
        data,
        &args.column,
        args.placeholder,
        args.filepath_placeholder,
        args.explain,
        95.0,
    )?;

    let suspect_ln = ModelHandler::compute_threshold(&model, 95.0);

    // Display results if not silent
    if !args.silent {
        ModelHandler::display_top(
            &results,
            &model,
            args.n_lines,
            args.color,
            args.show_percentage,
            args.explain,
            Some(suspect_ln),
        );
    }

    // Save results if requested
    if args.store || args.output.is_some() {
        ModelHandler::save_results(
            &results,
            args.output.as_deref(),
            &model,
            args.color,
            args.show_percentage,
            args.explain,
            Some(suspect_ln),
        )?;
    }

    Ok(())
}
