# anomark-rs (Python)

Python bindings for **anomark-rs** (this repository’s Rust implementation of AnoMark-style command-line anomaly detection). Load character or token Markov models and score command lines from Python.

## Build / install

From this directory (requires [Rust](https://rustup.rs/) and Python 3.9+):

```bash
pip install maturin
maturin develop --release
```

Or build a wheel:

```bash
maturin build --release
pip install target/wheels/anomark_rs-*.whl
```

## Usage

```python
import anomark_rs

m = anomark_rs.CharModel.load("models/my_model.bin")
print(m.order, m.prior)  # properties
score = m.score("/usr/bin/curl -s https://example.com")  # log-likelihood (higher = more typical)
print("suspect" if m.is_suspect(score) else "ok")

t = anomark_rs.TokenModel.load("models/token.bin")
print(t.score("curl http://example.com"))
```

See the main repository README for training models with the Rust CLI.
