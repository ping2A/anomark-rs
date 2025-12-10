use anyhow::Result;
use anomark::{apply_all_placeholders, load_csv, ModelHandler, train_parallel};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "train-from-csv")]
#[command(about = "Train a Markov model from CSV data", long_about = None)]
struct Args {
    /// Path of the CSV data to work on
    #[arg(short, long)]
    data: String,

    /// Name of the column in dataframe
    #[arg(short, long)]
    column: String,

    /// Model's order (number of letters for the window)
    #[arg(short, long)]
    order: usize,

    /// Count column name (if it exists)
    #[arg(long)]
    count_column: Option<String>,

    /// The path of the output for the new model
    #[arg(long)]
    output: Option<String>,

    /// The number of lines you want to take from csv
    #[arg(short = 'n', long)]
    n_lines: Option<usize>,

    /// The percentage of lines you want to take
    #[arg(short, long)]
    percentage: Option<f64>,

    /// Slice nLines from the end of dataset
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
    println!("Loading data from {}...", args.data);
    let mut data = load_csv(&args.data, &args.column)?;

    // Apply slicing
    if args.n_lines.is_some() || args.percentage.is_some() {
        data = anomark::process_data(
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

    // Load counts if provided
    let counts = if let Some(count_col) = &args.count_column {
        Some(load_csv(&args.data, count_col)?
            .into_iter()
            .filter_map(|s| s.parse::<usize>().ok())
            .collect::<Vec<_>>())
    } else {
        None
    };

    // Load existing model if resuming
    let existing_model = if args.resume {
        let model_path = args.model.as_ref()
            .expect("--model path required when using --resume");
        println!("Loading existing model from {}...", model_path);
        Some(ModelHandler::load_model(model_path)?)
    } else {
        None
    };

    let order = existing_model.as_ref()
        .map(|m| m.order)
        .unwrap_or(args.order);

    // Train model (parallel or serial)
    let mut model = if args.parallel {
        println!("Using parallel training mode...");
        train_parallel(&data, order, counts.as_deref(), existing_model)?
    } else {
        ModelHandler::train_from_csv(
            &data,
            order,
            counts.as_deref(),
            existing_model,
        )?
    };

    // Normalize model
    model.normalize_model_and_compute_prior();

    // Save model
    ModelHandler::save_model(&model, args.output.as_deref())?;

    Ok(())
}
