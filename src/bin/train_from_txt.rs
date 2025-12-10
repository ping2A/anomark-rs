use anyhow::Result;
use anomark::{apply_all_placeholders, load_txt, ModelHandler};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "train-from-txt")]
#[command(about = "Train a Markov model from TXT data", long_about = None)]
struct Args {
    /// Path of the TXT data to work on
    #[arg(short, long)]
    data: String,

    /// Model's order (number of letters for the window)
    #[arg(short, long)]
    order: usize,

    /// The path of the output for the new model
    #[arg(long)]
    output: Option<String>,

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
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load data
    println!("Loading data from {}...", args.data);
    let mut data = load_txt(&args.data)?;

    // Apply placeholders
    if args.placeholder {
        println!("Applying placeholder transformation...");
        data = apply_all_placeholders(&data, args.filepath_placeholder);
    }

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

    // Train model
    let mut model = ModelHandler::train_from_txt(
        &data,
        order,
        existing_model,
    )?;

    // Normalize model
    model.normalize_model_and_compute_prior();

    // Save model
    ModelHandler::save_model(&model, args.output.as_deref())?;

    Ok(())
}
