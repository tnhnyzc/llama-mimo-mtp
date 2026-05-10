# MiMo-V2.5 MTP Speculative Decoding

**Experimental. Tested on one hardware config (Apple M3 Max, Metal). Not benchmarked on dGPU. Things may be broken.**

Same-GGUF speculative decoding using MiMo-V2.5's built-in MTP heads as the draft model. No separate draft weights needed.

## What works

- p/q probability-based acceptance with stochastic draft sampling
- `--spec-draft-n-max 1` (single draft token per step) — tested, stable
- Apple M3 Max (Metal, 128 GB unified memory)

## What we don't know yet

- **dGPU performance.** Not tested. Should help given the draft is ~1% of target compute, but no benchmarks yet.
- **`--spec-draft-n-max 2+`.** Output is clean, but throughput is near baseline or slightly negative on M3 Max.

## Experimental knobs

Two flags for testing whether the BF16 draft signal can correct quantization noise in the target model. **Both default to off.**

- `--spec-logit-blend` (default: 0.0): Blends draft logits into target logits before softmax. On IQ3_S, we observed ~32% argmax disagreement between target and BF16 draft. With blend=2.0, acceptance goes from ~50% to ~94% on our hardware. Whether this matters at higher target precision or on dGPU is unknown.
- `--spec-accept-bias` (default: 1.0): Shifts the p/q acceptance threshold. >1.0 trusts the draft more.

Suggested starting point for experimentation: `--spec-logit-blend 2.0 --spec-accept-bias 4.0`

## Quick start

```bash
# Convert from HuggingFace (preserves MTP heads at BF16)
python3 convert_hf_to_gguf.py /path/to/MiMo-V2.5 \
  --outtype bf16 --mimo-mtp-layers 3 --outfile out.gguf

# Quantize (MTP tensors kept at BF16 automatically)
./build/bin/llama-quantize out.gguf out-IQ3_S.gguf IQ3_S

# Run
./build/bin/llama-server \
  --model out-IQ3_S.gguf \
  --spec-type mtp \
  --spec-draft-n-max 1 \
  -ngl 99
```

## Limitations

- MiMo2 architecture only
- `n_parallel` must be 1
- Requires GGUF converted with this fork's `convert_hf_to_gguf.py --mimo-mtp-layers 3` (upstream converter skips MTP tensors)
