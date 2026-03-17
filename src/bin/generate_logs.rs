//! Generate JSONL process-event logs for training or testing.
//! Output format: {"timestamp": "...", "event_type": "process", "user": "0", "command": "...", "pid": N, "ppid": M}

use anyhow::Result;
use anomark::LogGenerator;
use clap::Parser;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "generate-logs")]
#[command(about = "Generate JSONL process-event logs for training or regression testing", long_about = None)]
struct Args {
    /// Output JSONL file path (default: stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Number of normal (baseline) event lines to generate
    #[arg(short, long, default_value = "1000")]
    count: usize,

    /// Add default set of normal kernel/process commands (recommended for regression)
    #[arg(long)]
    default_normal: bool,

    /// Add an anomalous command line (for testing detection)
    #[arg(long)]
    anomalous: Option<String>,

    /// Extra normal command (can be repeated)
    #[arg(long)]
    command: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut gen = LogGenerator::new();

    if args.default_normal {
        gen = gen.with_default_normal();
    }
    for cmd in &args.command {
        gen = gen.add_normal(cmd);
    }
    if let Some(a) = &args.anomalous {
        gen = gen.add_anomalous(a);
    }
    if !args.default_normal && args.command.is_empty() {
        // No commands at all: use default so we emit something
        gen = gen.with_default_normal();
    }

    let mut writer: BufWriter<Box<dyn std::io::Write>> = if let Some(p) = &args.output {
        BufWriter::new(Box::new(File::create(p)?))
    } else {
        BufWriter::new(Box::new(std::io::stdout()))
    };

    gen.emit(&mut writer, args.count)?;
    writer.flush()?;

    if args.output.is_some() {
        eprintln!("Generated {} lines to {:?}", args.count + if args.anomalous.is_some() { 1 } else { 0 }, args.output);
    }

    Ok(())
}
