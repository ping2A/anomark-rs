# AnoMark - Rust

**ORIGINAL PROJECT: https://github.com/ANSSI-FR/AnoMark**
**Created with Claude.ai but supervised by a human (me apparently) JUST FOR FUN.**

*Anomaly detection in command lines with Markov chains*

A pure Rust implementation of the AnoMark algorithm for detecting malicious command lines using Markov Chains and n-grams.

## Features

- 🦀 **Pure Rust** - No Python dependencies, native performance
- 🚀 **Fast execution** - Leverages Rust's performance for large datasets
- 📊 **Progress tracking** - Visual progress bars for long-running operations
- 💾 **Binary serialization** - Efficient model storage with bincode
- 🎨 **Colored output** - Highlights anomalous characters in terminal
- 🔧 **CLI tools** - Train and apply from CSV, TXT, or JSONL
- **Explainability** - See which n-grams contributed to a low score (character or token level)
- **Token-level models** - Optional word/token n-gram models (e.g. by whitespace or path segments)

## Installation

### Prerequisites

- Rust 1.70 or later
- Cargo (comes with Rust)

### Building from Source

```bash
# Clone the repository
git clone <repo-url>
cd anomark-rust

# Build release binaries
cargo build --release

# Binaries will be in target/release/
```

## Quick Start

### 1. Train a Model from CSV

```bash
cargo run --release --bin train-from-csv -- \
    -d data/train_data.csv \
    -c CommandLine \
    -o 4 \
    --placeholder
```

### 2. Train a Model from TXT

```bash
cargo run --release --bin train-from-txt -- \
    -d data/train_data.txt \
    -o 4
```

### 3. Apply Model to Detect Anomalies

```bash
cargo run --release --bin apply-model -- \
    -m models/model.bin \
    -d data/test_data.csv \
    -c CommandLine \
    --store \
    --color \
    -n 100
```

## Command-Line Tools

### train-from-csv

Train a model from CSV data with optional data slicing and preprocessing.

```bash
train-from-csv [OPTIONS] --data <PATH> --column <NAME> --order <NUM>

Options:
    -d, --data <PATH>              Path to CSV file
    -c, --column <NAME>            Column name containing command lines
    -o, --order <NUM>              N-gram order (window size)
        --count-column <NAME>      Column containing occurrence counts
        --output <PATH>            Custom output path for model
    -n, --n-lines <NUM>            Number of lines to use for training
    -p, --percentage <PCT>         Percentage of data to use
        --from-end                 Slice from end of dataset
    -r, --randomize                Randomize line selection
        --placeholder              Apply GUID/SID/User/Hash placeholders
        --filepath-placeholder     Apply filepath placeholders
        --resume                   Continue training existing model
    -m, --model <PATH>             Path to existing model (with --resume)
```

**Examples:**

```bash
# Basic training
train-from-csv -d data.csv -c CommandLine -o 4

# Train on first 1000 lines with placeholders
train-from-csv -d data.csv -c CommandLine -o 4 -n 1000 --placeholder

# Train on 50% of data, randomized
train-from-csv -d data.csv -c CommandLine -o 4 -p 50 -r

# Resume training from existing model
train-from-csv -d new_data.csv -c CommandLine --resume -m models/existing.bin
```

### train-from-txt

Train a model from plain text data.

```bash
train-from-txt [OPTIONS] --data <PATH> --order <NUM>

Options:
    -d, --data <PATH>              Path to TXT file
    -o, --order <NUM>              N-gram order (window size)
        --output <PATH>            Custom output path for model
        --placeholder              Apply GUID/SID/User/Hash placeholders
        --filepath-placeholder     Apply filepath placeholders
        --resume                   Continue training existing model
    -m, --model <PATH>             Path to existing model (with --resume)
```

### apply-model

Apply a trained model to detect anomalies in new data.

```bash
apply-model [OPTIONS] --data <PATH> --model <PATH> --column <NAME>

Options:
    -d, --data <PATH>              Path to CSV file to analyze
    -m, --model <PATH>             Path to trained model
    -c, --column <NAME>            Column name to analyze
    -s, --store                    Save results to CSV
    -o, --output <PATH>            Custom output path
        --color                    Highlight anomalous characters
    -n, --n-lines <NUM>            Number of results to display [default: 50]
        --silent                   Suppress terminal output
        --placeholder              Apply placeholders to test data
        --filepath-placeholder     Apply filepath placeholders
        --show-percentage          Show anomaly percentage scores
```

**Examples:**

```bash
# Basic execution with colored output
apply-model -m models/model.bin -d test.csv -c CommandLine --color

# Save results and show top 100 anomalies
apply-model -m models/model.bin -d test.csv -c CommandLine -s -n 100

# Apply with placeholders and percentage scores
apply-model -m models/model.bin -d test.csv -c CommandLine \
    --placeholder --show-percentage --store

# Explain why a command is anomalous (show unusual n-grams)
apply-model -m models/model.bin -d test.csv -c CommandLine --explain -n 20
```

### Explainability

You can get **explanations** for why a command was scored as anomalous: the model reports which **unusual n-grams** (character or token sequences) had low probability and contributed to the low score.

- **Character model**: use `apply-model` with `--explain`. The CLI prints unusual character n-grams for each result, and the CSV export includes an **UnusualNgrams** column (semicolon-separated list of `ngram (log_prob)`).
- **Token model**: use `apply-token-model` with `--explain` for token-level explanations (e.g. which token transitions were rare).

**How it works**: The model scores each (order+1)-gram in the sequence. N-grams with log-probability below a threshold (e.g. 95% of the prior) are flagged as “unusual” and attached to the result. Lower log-probability means the transition was rare in training, so it contributes to anomaly.

**Example** (character model):

```bash
apply-model -m models/demo_char.bin -d data/demo_logs.jsonl -c command -n 10 --explain
```

Output includes lines like:

```
  unusual n-grams: "xyz"(-7.2), "ab"(-6.8), ...
```

### Token-level models

Besides **character n-grams**, you can train a **token-level** Markov model (e.g. over words or path segments). This can help when anomalies are better expressed as “unusual token sequences” rather than unusual character sequences.

**Train a token model** (from JSONL or CSV):

```bash
train-token-model -d data/commands.jsonl -c command -o 2 --tokenizer whitespace --output models/token.bin
```

**Tokenizer options**:

- `whitespace` – split on spaces (default)
- `path` – split on `/` and `\`, keeping separators as tokens
- `whitespace_and_path` – path split, then whitespace within segments

**Apply the token model**:

```bash
apply-token-model -m models/token.bin -d data/test.jsonl -c command -n 20 --explain
```

Token models use the same **explainability** as the character model: with `--explain`, results include unusual token transitions (e.g. `"curl -> http" (-5.1)`).

**Demo** (generates data, trains both character and token models, runs detection with explain):

```bash
./demo_explain_and_token.sh
```

## Placeholder Transformations

The `--placeholder` flag replaces common variable elements with placeholders to reduce false positives:

| Pattern | Regex | Placeholder |
|---------|-------|-------------|
| GUID | `\{?[0-9A-Fa-f]{8}[-–]([0-9A-Fa-f]{4}[-–]){3}[0-9A-Fa-f]{12}\}?` | `<GUID>` |
| SID | `S[-–]1[-–]([0-9]+[-–])+[0-9]+` | `<SID>` |
| User Path | `(C:\\Users)\\[^\\]*\\` | `<USER>` |
| Hash | `\b(?:[A-Fa-f0-9]{64}\|[A-Fa-f0-9]{40}\|[A-Fa-f0-9]{32}\|[A-Fa-f0-9]{20})\b` | `<HASH>` |

The `--filepath-placeholder` replaces full file paths with `<FILEPATH>`. Use cautiously as it may affect true positive detection.


## Library Usage

You can also use AnoMark as a Rust library:

```rust
use anomark::{ModelHandler, MarkovModel};

fn main() -> anyhow::Result<()> {
    // Train a model
    let training_data = "normal command line patterns...";
    let mut model = ModelHandler::train_from_txt(training_data, 4, None)?;
    model.normalize_model_and_compute_prior();
    
    // Score new data
    let test_text = "suspicious command";
    let score = model.log_likelihood(test_text);
    
    println!("Anomaly score: {}", score);
    Ok(())
}
```

## Performance Comparison

The Rust version offers significant performance improvements:

- **Training**: ~5-10x faster than Python
- **Execution**: ~3-5x faster than Python
- **Memory**: ~30-50% less memory usage
- **Binary size**: Compiled models are more compact

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_train_from_txt
```

## Development

```bash
# Check code
cargo check

# Format code
cargo fmt

# Lint code
cargo clippy

# Build documentation
cargo doc --open
```

## Algorithm Details

AnoMark uses character-level n-grams to build a Markov chain model:

1. **Training**: Learns transition probabilities between character sequences
2. **Normalization**: Converts counts to probabilities and computes prior
3. **Scoring**: Computes average log-likelihood for test sequences
4. **Detection**: Lower scores indicate more anomalous patterns

The `order` parameter determines the context window size (typically 3-5 characters).

## References

- Original Python implementation: [ANSSI-FR/AnoMark](https://github.com/ANSSI-FR/AnoMark)
- SSTIC 2022 Presentation (French): [Link](https://www.sstic.org/2022/presentation/anomark_detection_anomalies_dans_des_lignes_de_commande_chaines_de_markov/)
- FIRST 2023 Conference (English): [Video](https://www.youtube.com/watch?v=RACIZZMzI9I)

## Troubleshooting

**Issue**: Model file not found
```bash
# Ensure models directory exists
mkdir -p models
```

**Issue**: CSV parsing errors
```bash
# Check CSV format and column names
cargo run --release --bin apply-model -- --help
```

**Issue**: Out of memory during training
```bash
# Use line limiting flags
train-from-csv -d data.csv -c CommandLine -o 4 -n 10000
```
