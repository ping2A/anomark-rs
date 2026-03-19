//! Unified character-model training from CSV, JSONL, or plain text (format auto-detected or forced).

use anyhow::Result;
use anomark::{
    apply_all_placeholders,
    expand_train_files,
    load_char_training_data,
    maybe_filter_training_lines,
    maybe_filter_txt_training_body,
    process_data,
    resolve_column_name,
    train_parallel,
    validate_train_file_kinds,
    LoadedCharTrainingData,
    ModelHandler,
    TrainFileKind,
    TrainFormatArg,
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "train")]
#[command(
    about = "Train a character-level Markov model from CSV, JSONL, and/or .txt (auto-detect by extension, or set --format)",
    long_about = None
)]
struct Args {
    /// Input format: auto (from extension), csv, jsonl, or txt
    #[arg(long, value_enum, default_value_t = TrainFormatArg::Auto)]
    format: TrainFormatArg,

    /// Path(s) to data file(s) and/or directories (can repeat). With `auto`, directories pick up .csv, .jsonl, and .txt.
    #[arg(short, long, num_args = 1..)]
    data: Vec<String>,

    #[arg(long)]
    recursive: bool,

    /// Column (CSV) or JSON field (JSONL); required for CSV. For JSONL-only input, defaults to `command` if omitted.
    #[arg(short, long)]
    column: Option<String>,

    /// Only for JSONL: keep lines where FIELD equals VALUE (e.g. event_type process)
    #[arg(long, number_of_values = 2, value_names = &["FIELD", "VALUE"])]
    filter: Option<Vec<String>>,

    #[arg(short, long)]
    order: usize,

    #[arg(long)]
    count_column: Option<String>,

    #[arg(long)]
    output: Option<String>,

    #[arg(short = 'n', long)]
    n_lines: Option<usize>,

    #[arg(short, long)]
    percentage: Option<f64>,

    #[arg(long)]
    from_end: bool,

    #[arg(short, long)]
    randomize: bool,

    #[arg(long)]
    placeholder: bool,

    #[arg(long)]
    filepath_placeholder: bool,

    #[arg(long)]
    resume: bool,

    #[arg(short, long)]
    model: Option<String>,

    #[arg(long)]
    parallel: bool,

    #[arg(long)]
    exclude_kernel_threads: bool,

    #[arg(long = "exclude-regex", value_name = "PATTERN")]
    exclude_regex: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let files = expand_train_files(&args.data, args.recursive, args.format)?;
    if files.is_empty() {
        anyhow::bail!(
            "No training files found under path(s): {}",
            args.data.join(", ")
        );
    }
    validate_train_file_kinds(&files)?;

    let (filter_field, filter_value) = match &args.filter {
        Some(v) if v.len() == 2 => (Some(v[0].as_str()), Some(v[1].as_str())),
        _ => (None, None),
    };
    if files.iter().any(|(_, k)| *k == TrainFileKind::Jsonl) {
        if filter_field.is_some() {
            println!(
                "JSONL filter: {} = {}",
                filter_field.unwrap(),
                filter_value.unwrap()
            );
        }
    } else if args.filter.is_some() {
        anyhow::bail!("--filter applies only to JSONL input");
    }

    if args.count_column.is_some()
        && !files
            .iter()
            .any(|(_, k)| *k == TrainFileKind::Csv)
    {
        anyhow::bail!("--count-column is only used with CSV files");
    }

    let column = resolve_column_name(&files, args.column.as_deref())?;
    println!("Loading {} training file(s)...", files.len());

    let loaded = load_char_training_data(
        &files,
        column.as_deref(),
        filter_field,
        filter_value,
        args.count_column.as_deref(),
    )?;

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

    match loaded {
        LoadedCharTrainingData::Lines { lines, counts } => {
            let (lines, counts, excluded_train) = maybe_filter_training_lines(
                lines,
                counts,
                args.exclude_kernel_threads,
                &args.exclude_regex,
            )?;
            if excluded_train > 0 {
                println!(
                    "Excluded {} command(s) from training (--exclude-kernel-threads / --exclude-regex)",
                    excluded_train
                );
            }
            if lines.is_empty() {
                anyhow::bail!("No training lines left after exclusions; relax filters or check input.");
            }

            let mut lines = lines;
            if args.n_lines.is_some() || args.percentage.is_some() {
                lines = process_data(
                    lines,
                    args.n_lines,
                    args.percentage,
                    args.from_end,
                    args.randomize,
                );
            }

            if args.placeholder {
                println!("Applying placeholder transformation...");
                lines = lines
                    .into_iter()
                    .map(|text| apply_all_placeholders(&text, args.filepath_placeholder))
                    .collect();
            }

            let mut model = if args.parallel {
                println!("Using parallel training mode...");
                train_parallel(&lines, order, counts.as_deref(), existing_model)?
            } else {
                ModelHandler::train_from_csv(&lines, order, counts.as_deref(), existing_model)?
            };
            model.normalize_model_and_compute_prior();
            ModelHandler::save_model(&model, args.output.as_deref())?;
        }
        LoadedCharTrainingData::Corpus { text } => {
            let (mut text, excluded_train) = maybe_filter_txt_training_body(
                text,
                args.exclude_kernel_threads,
                &args.exclude_regex,
            )?;
            if excluded_train > 0 {
                println!(
                    "Excluded {} line(s) from training (--exclude-kernel-threads / --exclude-regex)",
                    excluded_train
                );
            }
            if text.trim().is_empty() {
                anyhow::bail!("No training text left after exclusions; relax filters or check input.");
            }

            if args.n_lines.is_some() || args.percentage.is_some() {
                let as_lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
                let sliced = process_data(
                    as_lines,
                    args.n_lines,
                    args.percentage,
                    args.from_end,
                    args.randomize,
                );
                text = sliced.join("\n");
                if !text.is_empty() {
                    text.push('\n');
                }
            }

            if args.placeholder {
                println!("Applying placeholder transformation...");
                text = apply_all_placeholders(&text, args.filepath_placeholder);
            }

            let mut model = ModelHandler::train_from_txt(&text, order, existing_model)?;
            model.normalize_model_and_compute_prior();
            ModelHandler::save_model(&model, args.output.as_deref())?;
        }
    }

    Ok(())
}
