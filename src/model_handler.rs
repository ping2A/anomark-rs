use crate::data_handler::{apply_all_placeholders, load_csv_with_columns};
use crate::model::MarkovModel;
use ahash::AHashMap;
use anyhow::{Context, Result};
use chrono::Local;
use csv::Writer;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::{BufReader, BufWriter};

const MARKOV_SCORE: &str = "markovScore";

pub struct ModelHandler;

impl ModelHandler {
    /// Train a model from text data
    pub fn train_from_txt(
        training_data: &str,
        model_order: usize,
        model: Option<MarkovModel>,
    ) -> Result<MarkovModel> {
        let mut model = model.unwrap_or_else(|| MarkovModel::new(model_order));
        
        println!("Training model...");
        let start = std::time::Instant::now();
        
        model.train(training_data, 1);
        
        println!("Training took {:.2} minutes", start.elapsed().as_secs_f64() / 60.0);
        
        Ok(model)
    }

    /// Train a model from CSV data
    pub fn train_from_csv(
        data: &[String],
        model_order: usize,
        count_data: Option<&[usize]>,
        model: Option<MarkovModel>,
    ) -> Result<MarkovModel> {
        let mut model = model.unwrap_or_else(|| MarkovModel::new(model_order));
        
        println!("Training model...");
        let start = std::time::Instant::now();
        
        let pb = ProgressBar::new(data.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        for (i, text) in data.iter().enumerate() {
            let padded = format!("{}{}{}", "~".repeat(model_order), text, "~".repeat(model_order));
            let count = count_data.map(|c| c[i]).unwrap_or(1);
            model.train(&padded, count);
            pb.inc(1);
        }

        pb.finish_with_message("Training complete");
        println!("Training took {:.2} minutes", start.elapsed().as_secs_f64() / 60.0);
        
        Ok(model)
    }

    /// Save a model to disk
    pub fn save_model(model: &MarkovModel, save_path: Option<&str>) -> Result<String> {
        println!("Saving model...");
        
        let path = if let Some(p) = save_path {
            p.to_string()
        } else {
            let now = Local::now();
            format!(
                "./models/{}_modelLetters_{}grams.bin",
                now.format("%Y%m%d_%Hh%M"),
                model.order
            )
        };

        std::fs::create_dir_all("./models")?;

        let file = File::create(&path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, model)
            .context("Failed to serialize model")?;

        println!("Successfully saved model in: {}", path);
        Ok(path)
    }

    /// Load a model from disk
    pub fn load_model(path: &str) -> Result<MarkovModel> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model: MarkovModel = bincode::deserialize_from(reader)
            .context("Failed to deserialize model")?;
        Ok(model)
    }

    /// Execute model on dataset and return scored results
    pub fn execute_on_data(
        model: &mut MarkovModel,
        data: Vec<AHashMap<String, String>>,
        col_name: &str,
        apply_placeholder: bool,
        apply_filepath: bool,
    ) -> Result<Vec<ScoredResult>> {
        if !model.is_trained() {
            model.normalize_model_and_compute_prior();
        }

        println!("Applying model to data...");
        
        let pb = ProgressBar::new(data.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut results = Vec::new();
        let mut grouped: AHashMap<String, ScoredResult> = AHashMap::new();

        for row in data {
            let mut text = row.get(col_name)
                .context("Column not found")?
                .clone();

            if apply_placeholder {
                text = apply_all_placeholders(&text, apply_filepath);
            }

            let padded = format!("{}{}", "~".repeat(model.order), text);
            let score = model.log_likelihood(&padded);

            let entry = grouped.entry(text.clone()).or_insert_with(|| ScoredResult {
                command_line: text,
                score,
                other_fields: AHashMap::new(),
            });

            // Aggregate other columns
            for (key, value) in row {
                if key != col_name {
                    entry.other_fields
                        .entry(key)
                        .or_insert_with(Vec::new)
                        .push(value);
                }
            }

            // Keep minimum score for this command line
            if score < entry.score {
                entry.score = score;
            }

            pb.inc(1);
        }

        pb.finish_with_message("Execution complete");

        results = grouped.into_values().collect();
        results.sort_by(|a, b| {
            a.score.partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Compute threshold as percentage of model prior
    pub fn compute_threshold(model: &MarkovModel, percent: f64) -> f64 {
        model.prior.ln() * percent / 100.0
    }

    /// Add colored output to a command line
    pub fn colored_results(text: &str, model: &MarkovModel, threshold: f64) -> String {
        let padded = format!("{}{}", "~".repeat(model.order), text);
        let chars: Vec<char> = padded.chars().collect();
        let mut result = String::new();

        for k in model.order..chars.len() {
            let ngram: String = chars[k.saturating_sub(model.order)..=k].iter().collect();
            let score = model.log_likelihood(&ngram);
            
            if score < threshold {
                result.push_str(&format!("\x1b[91m{}\x1b[0m", chars[k]));
            } else {
                result.push(chars[k]);
            }
        }

        result.replace("\x1b[0m\x1b[91m", "")
    }

    /// Display top results in terminal
    pub fn display_top(
        results: &[ScoredResult],
        model: &MarkovModel,
        nb_lines: usize,
        color: bool,
        show_percentage: bool,
    ) {
        println!("_______");
        println!("Displaying top {}", nb_lines);

        let threshold = Self::compute_threshold(model, 95.0);

        for result in results.iter().take(nb_lines) {
            println!("_______");
            
            if color {
                println!("{}", Self::colored_results(&result.command_line, model, threshold));
            } else {
                println!("{}", result.command_line);
            }

            if show_percentage {
                let percentage = (result.score / model.prior.ln() * 100.0).round();
                println!("{}%", percentage);
            }
        }
        println!("_______");
    }

    /// Save results to CSV file
    pub fn save_results(
        results: &[ScoredResult],
        output: Option<&str>,
        model: &MarkovModel,
        color: bool,
        show_percentage: bool,
    ) -> Result<String> {
        println!("Saving results...");
        
        let path = if let Some(p) = output {
            p.to_string()
        } else {
            let now = Local::now();
            format!("./results/{}_export.csv", now.format("%Y%m%d_%Hh%M"))
        };

        std::fs::create_dir_all("./results")?;

        let mut wtr = Writer::from_path(&path)?;

        // Write headers
        let mut headers = vec!["CommandLine".to_string(), MARKOV_SCORE.to_string()];
        if let Some(first) = results.first() {
            for key in first.other_fields.keys() {
                headers.push(format!("List of all {}", key));
            }
        }
        if color {
            headers.push("Colored CommandLine".to_string());
        }
        if show_percentage {
            headers.push("Percentage".to_string());
        }
        wtr.write_record(&headers)?;

        let threshold = Self::compute_threshold(model, 95.0);

        // Write data
        for result in results {
            let mut record = vec![
                result.command_line.clone(),
                result.score.to_string(),
            ];

            for key in headers.iter().skip(2) {
                if key.starts_with("List of all ") {
                    let original_key = key.trim_start_matches("List of all ");
                    if let Some(values) = result.other_fields.get(original_key) {
                        let unique: Vec<String> = values.iter()
                            .map(|s| s.to_string())
                            .collect::<std::collections::HashSet<_>>()
                            .into_iter()
                            .collect();
                        record.push(unique.join(" - "));
                    } else {
                        record.push(String::new());
                    }
                }
            }

            if color {
                record.push(Self::colored_results(&result.command_line, model, threshold));
            }

            if show_percentage {
                let percentage = (result.score / model.prior.ln() * 100.0).round();
                record.push(format!("{}%", percentage));
            }

            wtr.write_record(&record)?;
        }

        wtr.flush()?;
        println!("Successfully saved results in: {}", path);
        Ok(path)
    }
}

#[derive(Debug, Clone)]
pub struct ScoredResult {
    pub command_line: String,
    pub score: f64,
    pub other_fields: AHashMap<String, Vec<String>>,
}
