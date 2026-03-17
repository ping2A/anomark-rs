use anyhow::Result;
use anomark::{apply_all_placeholders, load_jsonl, ModelHandler, process_data, train_parallel};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "train-from-jsonl")]
#[command(about = "Train a Markov model from JSONL data (e.g. process events with command field)", long_about = None)]
struct Args {
    /// Path of the JSONL data file
    #[arg(short, long)]
    data: String,

    /// Name of the JSON field to use as training text (e.g. "command")
    #[arg(short, long, default_value = "command")]
    column: String,

    /// Model's order (number of letters for the window)
    #[arg(short, long)]
    order: usize,

    /// The path of the output for the new model
    #[arg(long)]
    output: Option<String>,

    /// The number of lines you want to take from the file
    #[arg(short = 'n', long)]
    n_lines: Option<usize>,

    /// The percentage of lines you want to take
    #[arg(short, long)]
    percentage: Option<f64>,

    /// Slice n_lines from the end of dataset
    #[arg(long)]
    from_end: bool,

    /// Randomize selection for line selection
    #[arg(short, long)]
    randomize: bool,

    /// Apply GUID, SID, username, and hash replacement by placeholder
    #[arg(long)]
    placeholder: bool,

    /// Apply filepath replacement by placeholder
    #[arg(long)]
    filepath_placeholder: bool,

    /// Continue training mode for the model
    #[arg(long)]
    resume: bool,

    /// Path to the model to use (resume training mode)
    #[arg(short, long)]
    model: Option<String>,

    /// Use parallel processing (faster for large datasets)
    #[arg(long)]
    parallel: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load data
    println!("Loading data from {} (field: {})...", args.data, args.column);
    let mut data = load_jsonl(&args.data, &args.column)?;
    println!("Loaded {} records", data.len());

    // Apply slicing
    if args.n_lines.is_some() || args.percentage.is_some() {
        data = process_data(
            data,
            args.n_lines,
            args.percentage,
            args.from_end,
            args.randomize,
        );
    }

    // Apply placeholders
    if args.placeholder {
        println!("Applying placeholder transformation...");
        data = data
            .into_iter()
            .map(|text| apply_all_placeholders(&text, args.filepath_placeholder))
            .collect();
    }

    // Load existing model if resuming
    let existing_model = if args.resume {
        let model_path = args
            .model
            .as_ref()
            .expect("--model path required when using --resume");
        println!("Loading existing model from {}...", model_path);
        Some(ModelHandler::load_model(model_path)?)
    } else {
        None
    };

    let order = existing_model
        .as_ref()
        .map(|m| m.order)
        .unwrap_or(args.order);

    // Train model (parallel or serial)
    let mut model = if args.parallel {
        println!("Using parallel training mode...");
        train_parallel(&data, order, None, existing_model)?
    } else {
        ModelHandler::train_from_csv(&data, order, None, existing_model)?
    };

    // Normalize model
    model.normalize_model_and_compute_prior();

    // Save model
    ModelHandler::save_model(&model, args.output.as_deref())?;

    Ok(())
}
