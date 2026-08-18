# Transformer benchmark

Flat scripts for timing, profiling, and memory-measuring a Transformer LM.

```
benchmarking.py
model.py
optimizer.py
nn_utils.py
data.py
```

## Setup

```sh
uv sync
```

`package = false` in `pyproject.toml` so `uv` does not build or install this repo as a package. The only declared deps are what the scripts import: `torch`, `numpy`, `einops`, `einx`, `jaxtyping`.

## Run

```sh
uv run python benchmarking.py small 512 5 5 full fp32 timing
```

Args: `model_name context_length warmup_steps measured_steps mode precision measurement`.
