# AnoMark - Rust

**ORIGINAL PROJECT: https://github.com/ANSSI-FR/AnoMark**  
**Created with Claude.ai but supervised by a human (me apparently) JUST FOR FUN.**

*Anomaly detection in command lines with Markov chains*

A pure Rust implementation of the AnoMark algorithm for detecting malicious command lines using Markov Chains and n-grams.

## This repository vs official AnoMark (ANSSI)

| | **Official [AnoMark](https://github.com/ANSSI-FR/AnoMark)** (ANSSI) | **This project (anomark-rs)** |
|---|--------|--------|
| **Language** | Python (notebooks + scripts) | Rust |
| **Runtime** | pip / conda, typical data-science stack | Native binaries + optional **Python bindings** ([PyO3](https://pyo3.rs/)) |
| **Training inputs** | CSV and text helpers in the upstream workflow | Unified **`train`** CLI: CSV, JSONL, TXT; directories; `--format`; token trainer separate |
| **Scoring / apply** | `apply_model.py` style workflow | `apply-model` / `apply-token-model`: JSONL, CSV, explainability, machine column, exclusions |
| **Model format** | Python pickle / project-specific | `bincode` `.bin` (character and token models) — **not interchangeable** with upstream pickles |
| **Extras here** | — | Token-level models, n-gram explainability, kernel-thread / regex exclusions, parallel training, mmap helpers, log generator, regression tests |

The upstream tool is the reference from ANSSI; this repo is an **independent Rust port** with a similar statistical idea (Markov / n-grams on command text) but **different code, CLI, and file formats**. For production use of the original behaviour and ecosystem, use the official project; use this repo when you want Rust performance, a single static binary workflow, or Python access via `anomark_rs` after building the bindings.

## Features

- 🦀 **Pure Rust** - No Python dependencies, native performance
- 🚀 **Fast execution** - Leverages Rust's performance for large datasets
- 📊 **Progress tracking** - Visual progress bars for long-running operations
- 💾 **Binary serialization** - Efficient model storage with bincode
- 🎨 **Colored output** - Highlights anomalous characters in terminal
- 🔧 **CLI tools** - One `train` command for CSV / JSONL / TXT character models; `apply-model`, token tools, etc.
- **Explainability** - See which n-grams contributed to a low score (character or token level)
- **Token-level models** - Optional word/token n-gram models (e.g. by whitespace or path segments)
- **Training exclusions** - Drop Linux kernel-style names (`[nvme-wq]`) and/or custom regex patterns before training
- **Python bindings** - Optional `anomark_rs` module (PyO3) to load models and score commands from Python

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

## Python bindings (`anomark_rs`)

The extension lives under [`bindings/anomark-py/`](bindings/anomark-py/). Build with [maturin](https://www.maturin.rs/) (recommended) or `cargo` from that directory.

```bash
cd bindings/anomark-py
pip install maturin
maturin develop --release   # editable install for the current venv
# or: maturin build --release && pip install ../../target/wheels/anomark_rs-*.whl
```

Example:

```python
import anomark_rs

m = anomark_rs.CharModel.load("models/my_model.bin")
s = m.score("/usr/bin/curl -s https://example.com")
print("suspect" if m.is_suspect(s) else "ok", s)

t = anomark_rs.TokenModel.load("models/token.bin")
print(t.score("curl http://example.com"))
```

API summary: **`CharModel.load` / `TokenModel.load`**, **`.score(command)`**, **`.suspect_threshold()`**, **`.is_suspect(score)`**, **`.explain(..., threshold_percent)`**, **`is_kernel_thread(cmd)`**, **`__version__`**. Train models with the Rust `train` / `train-token-model` CLIs first; `.bin` files are not compatible with upstream Python pickles.

On **macOS**, if `cargo build` alone fails to link, use `maturin` or the `rustflags` in [`bindings/anomark-py/.cargo/config.toml`](bindings/anomark-py/.cargo/config.toml).

## Quick Start

### 1. Train a character-level model (CSV, JSONL, or TXT — one binary)

Format is **auto-detected** from each file’s extension (`.csv`, `.jsonl`, `.txt`). You can force a format with `--format csv|jsonl|txt` (e.g. a `.log` file as JSONL). **Do not mix `.txt` with CSV/JSONL in the same run** (train those separately).

```bash
# CSV
cargo run --bin train -- -d data/train_data.csv -c CommandLine -o 4 --placeholder

# Plain text (one corpus per file, concatenated)
cargo run --bin train -- -d data/train_data.txt -o 4

# JSONL (process events; `-c` defaults to `command` if omitted for JSONL-only input)
cargo run --bin train -- \
    -d data/events.jsonl \
    -c command \
    -o 3 \
    --filter event_type process \
    --output models/process_model.bin
```

### 2. Apply Model to Detect Anomalies

```bash
cargo run --bin apply-model -- \
    -m models/model.bin \
    -d data/test_data.csv \
    -c CommandLine \
    --store \
    --color \
    -n 100
```

### Excluding commands from training

Linux and similar systems often report kernel threads with a **command line that is only a bracketed name**, e.g. `[kthreadd]`, `[nvme-wq]`. Those strings are usually not useful for learning “normal” userland commands.

The `train` command and `train-token-model` support:

| Flag | Meaning |
|------|--------|
| `--exclude-kernel-threads` | Drop lines where the **entire** command (after trim) matches `[something]` — one pair of brackets, no nested `[`/`]` inside. |
| `--exclude-regex <PATTERN>` | Drop lines matching this Rust regex (repeatable). Checked against the **full** command string (CSV/JSONL/token) or **each line** (TXT). |

Filtering runs **after** loading and **before** `-n` / `-p` slicing (so limits apply to the kept lines). If everything is excluded, the tool exits with an error. **When applying** the model, use the same `--exclude-kernel-threads` and/or `--exclude-regex` with `apply-model` (or `apply-token-model`) so those rows are skipped and not reported as anomalies.

```bash
# JSONL: train on process commands but skip kernel thread names
cargo run --bin train -- \
    -d data/events.jsonl -c command -o 3 \
    --filter event_type process \
    --exclude-kernel-threads \
    --output models/process_userland.bin

# CSV: kernel threads plus any extra pattern (e.g. lines containing "kworker")
cargo run --bin train -- \
    -d data.csv -c CommandLine -o 4 \
    --exclude-kernel-threads \
    --exclude-regex 'kworker'
```

## Command-Line Tools

### train

Train a **character-level** Markov model from **CSV**, **JSONL**, and/or **`.txt`** in one CLI. With `--format auto` (default), each file’s type is inferred from its extension; directories collect matching files (under `auto`, all of `.csv`, `.jsonl`, and `.txt` in that tree). **CSV and JSONL can be combined** in one run (same `-c` field name). **Plain `.txt` cannot be mixed** with CSV/JSONL in the same invocation.

```bash
cargo run --bin train -- [OPTIONS] --data <PATH>... --order <NUM>

Options:
        --format <FMT>             auto | csv | jsonl | txt [default: auto]
    -d, --data <PATH>...           File(s) and/or directories (repeatable); use --recursive for subdirs
        --recursive
    -c, --column <NAME>            CSV/JSONL field (required for CSV; JSONL-only defaults to command)
        --filter <FIELD> <VALUE>    JSONL only: keep lines where field equals value
    -o, --order <NUM>              N-gram order
        --count-column <NAME>      CSV only: per-row counts
        --output <PATH>
    -n, --n-lines <NUM>            Subsample lines (after exclusions)
    -p, --percentage <PCT>
        --from-end, -r, --randomize
        --placeholder, --filepath-placeholder
        --resume, -m, --model
        --parallel
        --exclude-kernel-threads, --exclude-regex <PATTERN>
```

**Examples:**

```bash
cargo run --bin train -- -d data.csv -c CommandLine -o 4
cargo run --bin train -- -d data.csv -c CommandLine -o 4 -n 1000 --placeholder
cargo run --bin train -- -d data/ -c CommandLine -o 4 --recursive
cargo run --bin train -- -d data/commands.txt -o 4
cargo run --bin train -- -d events.jsonl -o 3 --filter event_type process --output models/proc.bin
# Force JSONL for a file without .jsonl extension:
cargo run --bin train -- --format jsonl -d events.log -c command -o 3
# Merge two CSVs
cargo run --bin train -- -d a.csv -d b.csv -c CommandLine -o 4
```

### apply-model

Apply a trained model to detect anomalies in new data. Input can be **CSV or JSONL**; format is auto-detected from the file extension (`.jsonl` → JSONL) or set with `--format`.

**Mixed JSONL (different fields per line):** If your JSONL has different event types (e.g. some lines with `date`, `file_path`, `event_type` and others with `command`, `pid`), the tool scans all rows to find the first one that has the requested column. Rows that don't have that field (e.g. file events when you use `-c command`) are **skipped**; at the end you'll see e.g. `Skipped N rows missing field 'command' (e.g. other event types)`.

```bash
cargo run --bin apply-model -- [OPTIONS] --data <PATH> --model <PATH> --column <NAME>

Options:
    -d, --data <PATH>              Path to CSV or JSONL file to analyze
    -m, --model <PATH>             Path to trained model
    -c, --column <NAME>            Column/field to score (e.g. CommandLine for CSV, command for JSONL)
        --format <FMT>             Input format: csv, jsonl, or auto [default: auto]
    -s, --store                    Save results to CSV
    -o, --output <PATH>            Custom output path
        --color                    Highlight anomalous characters
    -n, --n-lines <NUM>            Number of results to display [default: 50]
        --silent                   Suppress terminal output
        --placeholder              Apply placeholders to test data
        --filepath-placeholder     Apply filepath placeholders
        --show-percentage          Show anomaly percentage scores
        --explain                  Show unusual n-grams for each result
        --exclude-kernel-threads   Skip rows whose command is [name]-style (e.g. [kthreadd]); use if you trained with --exclude-kernel-threads
        --exclude-regex <PATTERN>  Skip rows whose command matches this regex (repeatable)
        --machine-field <COLUMN>   Column/field that contains the machine/host name (e.g. hostname); output includes a Machine column for filtering
        --machine <NAME>          Use this value as Machine for every row (e.g. when input has no host column)
```

**Machine / host for filtering:** Use `--machine-field hostname` (or your column name) so the CSV and terminal output include a **Machine** column; you can then filter or group by machine. If the input has no host column, use `--machine server01` to tag all rows from this run.

**Apply-time exclusions:** If you trained with `--exclude-kernel-threads` (or `--exclude-regex`), use the same flags when applying so those rows are **skipped** and not reported as anomalies. Otherwise kernel threads (e.g. `[nvme-wq]`) will appear with low scores because they were never in the training set.

**Suspect commands:** Each printed line is labeled **`SUSPECT (this command is flagged as unusual)`** or **`not flagged`** using `markovScore` vs the model baseline (95% of prior log-probability). Exported CSV includes a **`Suspect`** column (`yes` / `no`) right after `markovScore`. Results stay sorted with the **most unusual first** (`#1`).

**Examples (CSV):**

```bash
# Basic execution with colored output
cargo run --bin apply-model -- -m models/model.bin -d test.csv -c CommandLine --color

# Save results and show top 100 anomalies
cargo run --bin apply-model -- -m models/model.bin -d test.csv -c CommandLine -s -n 100

# Apply with placeholders and percentage scores
cargo run --bin apply-model -- -m models/model.bin -d test.csv -c CommandLine \
    --placeholder --show-percentage --store

# Explain why a command is anomalous (show unusual n-grams)
cargo run --bin apply-model -- -m models/model.bin -d test.csv -c CommandLine --explain -n 20
```

**Examples (JSONL):**

```bash
# Run on JSONL (e.g. process events); format auto-detected from .jsonl extension
cargo run --bin apply-model -- -m models/process_model.bin -d data/events.jsonl -c command -n 20

# Force JSONL when file has no .jsonl extension
cargo run --bin apply-model -- -m models/process_model.bin -d data/events.log --format jsonl -c command

# JSONL with explain and store
cargo run --bin apply-model -- -m models/process_model.bin -d data/events.jsonl -c command --explain -s -o results/anomalies.csv
```

### Explainability

You can get **explanations** for why a command was scored as anomalous: the model reports which **unusual n-grams** (character or token sequences) had low probability and contributed to the low score.

- **Character model**: use `apply-model` with `--explain`. The CLI prints unusual character n-grams for each result, and the CSV export includes an **UnusualNgrams** column (semicolon-separated list of `ngram (log_prob)`).
- **Token model**: use `apply-token-model` with `--explain` for token-level explanations (e.g. which token transitions were rare).

**How it works**: The model scores each (order+1)-gram in the sequence. N-grams with log-probability below a threshold (e.g. 95% of the prior) are flagged as “unusual” and attached to the result. Lower log-probability means the transition was rare in training, so it contributes to anomaly.

**Example** (character model):

```bash
cargo run --bin apply-model -- -m models/demo_char.bin -d data/demo_logs.jsonl -c command -n 10 --explain
```

Output includes lines like:

```
  unusual n-grams: "xyz"(-7.2), "ab"(-6.8), ...
```

### Token-level models

Besides **character n-grams**, you can train a **token-level** Markov model (e.g. over words or path segments). This can help when anomalies are better expressed as “unusual token sequences” rather than unusual character sequences.

**Train a token model** (from JSONL or CSV). Same `--exclude-kernel-threads` and `--exclude-regex` as other trainers:

```bash
cargo run --bin train-token-model -- -d data/commands.jsonl -c command -o 2 --tokenizer whitespace --output models/token.bin

cargo run --bin train-token-model -- -d data/commands.jsonl -c command -o 2 --exclude-kernel-threads --output models/token.bin
```

**Tokenizer options**:

- `whitespace` – split on spaces (default)
- `path` – split on `/` and `\`, keeping separators as tokens
- `whitespace_and_path` – path split, then whitespace within segments

**Apply the token model**:

```bash
cargo run --bin apply-token-model -- -m models/token.bin -d data/test.jsonl -c command -n 20 --explain
```

Token models use the same **explainability** as the character model: with `--explain`, results include unusual token transitions (e.g. `"curl -> http" (-5.1)`).

**Demo** (generates data, trains both character and token models, runs detection with explain):

```bash
./demo_explain_and_token.sh
```

### inspect-model

Show **summary statistics** for a saved model (character or token). The tool tries to load as a character model first, then as a token model.

```bash
cargo run --bin inspect-model -- -m models/my_model.bin
```

Human-readable output includes: file size, model type, order, prior, whether the chain is trained, number of context n-grams, transition count, and (for character models) alphabet size.

**JSON** (for scripts / dashboards):

```bash
cargo run --bin inspect-model -- -m models/my_model.bin --json
```

From Rust you can also use `MarkovModel::num_contexts()`, `num_transitions()`, `alphabet_len()`, and the same on `TokenMarkovModel` (`num_contexts`, `num_transitions`) after loading with `ModelHandler::load_model` / `load_token_model`.

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

**Issue**: `Column/field '…' not found` (or old message `Column not found`) when running `apply-model`

- **JSONL**: Field names are **case-sensitive** in the file, but `-c` / `--column` is matched **case-insensitively** (e.g. `-c Command` matches `"command"`). If it still fails, the error lists **available fields** from the first row—use one of those names exactly.
- **Wrong format**: If the path does not end in `.jsonl` but the file is JSONL, add `--format jsonl`.
- **CSV**: Use the exact header name from the first line (whitespace matters); case-insensitive matching also applies.

**Issue**: Model file not found
```bash
# Ensure models directory exists
mkdir -p models
```

**Issue**: CSV parsing errors
```bash
# Check CSV format and column names
cargo run --bin apply-model -- --help
```

**Issue**: Out of memory during training
```bash
# Use line limiting flags
cargo run --bin train -- -d data.csv -c CommandLine -o 4 -n 10000
```
