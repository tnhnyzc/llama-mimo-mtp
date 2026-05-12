# MiMo-V2.5 MTP Speculative Decoding

Experimental same-GGUF MTP speculative decoding for MiMo-V2.5 in llama.cpp.

This fork loads the normal MiMo target model and an internal `mimo2_mtp` draft head from the same GGUF. No separate draft model file is needed.

## Current State

Tested locally:

- Apple M3 Max, Metal, 128 GB unified memory.
- MiMo-V2.5 IQ3_S-style GGUF with the main model quantized and core `nextn` tensors kept high precision.
- `llama-server --spec-type mtp --spec-draft-n-max 1`.

Works:

- MiMo-V2.5 conversion with appended MTP / `nextn` tensors.
- MiMo target trunk skips appended MTP layers during normal inference.
- `mimo2_mtp` draft architecture loads the appended MTP layers from the same GGUF.
- p/q speculative acceptance for stochastic sampling.
- Conservative `nmax=1` serving.

Not proven:

- dGPU performance.
- `nmax=2+` as a useful speed path.
- File/slot prompt-cache save/restore behavior with MTP.

## Quick Start

```bash
# Convert from a MiMo-V2.5 HF directory that includes model_mtp.safetensors.
python3 convert_hf_to_gguf.py /path/to/MiMo-V2.5 \
  --outtype bf16 \
  --mimo-mtp-layers 3 \
  --outfile /path/to/MiMo-V2.5-MTP3-BF16.gguf

# Quantize. This fork keeps core nextn projection/norm tensors high precision.
./build/bin/llama-quantize \
  /path/to/MiMo-V2.5-MTP3-BF16.gguf \
  /path/to/MiMo-V2.5-MTP3-IQ3_S.gguf \
  IQ3_S

# Run the tested conservative mode.
./build/bin/llama-server \
  --model /path/to/MiMo-V2.5-MTP3-IQ3_S.gguf \
  --spec-type mtp \
  --spec-draft-n-max 1 \
  -np 1 \
  -ngl 99
```

Add your normal sampling/server args as needed. The current tested path is server-first; CLI behavior has not been the focus.

## Runtime Constraints

Required:

- `general.architecture = mimo2`.
- GGUF converted with this fork, not upstream llama.cpp, because upstream conversion does not export MiMo MTP tensors.
- `mimo2.nextn_predict_layers > 0`.
- `-np 1` / `--parallel 1`. The server rejects `n_parallel > 1` for MTP.

Automatically handled:

- The internal MTP draft model is loaded with `override_arch = mimo2_mtp`.
- The internal MTP draft load uses `use_mmap = false` so it does not mmap the whole target GGUF just to load the small draft subset.
- `--cache-reuse` KV-shift reuse and context shift are disabled when MTP is active.

Recommended:

- Start with `--spec-draft-n-max 1`.
- Use `--flash-attn auto` or `--flash-attn on` on backends where it is supported.
- Keep core `nextn` tensors high precision when quantizing small target quants such as IQ3_S.

Avoid for normal serving:

- `--spec-draft-n-max 2` or higher, unless you are explicitly experimenting.
- `LLAMA_MIMO_MTP_TREE_*` flags.
- `LLAMA_MIMO_MTP_MULTI_CTX=1`, unless testing the multi-layer/tree branch behavior.
- `--cache-reuse` / context-shift assumptions.

## Experimental Sampling Knobs

These default to conservative/off values.

- `--spec-calib-temp`: draft calibration temperature for p/q acceptance.
- `--spec-accept-bias`: shifts p/q acceptance. Values above `1.0` trust the draft more and are heuristic.
- `--spec-logit-blend`: blends draft logits into target logits before p/q acceptance. This can increase acceptance on low-bit target quants, but it is no longer pure target distribution sampling.
- `--spec-garbage-thresh`: rejects very low draft-probability tokens.
- `--spec-dist-restore`: blends draft probability distribution into target probabilities after softmax.

Suggested experiment only:

```bash
--spec-logit-blend 2.0 --spec-accept-bias 4.0
```

Use the defaults if you want the least surprising behavior.

## Expected Performance

On the local M3 Max setup, `nmax=1` has shown prompt-dependent gains. Easy/high-agreement prompts can improve around 10-20%; lower-agreement prompts can be near baseline or slightly worse.

Fresh pause-point smoke, 64 generated tokens, `ctx=2048`. Treat absolute tok/s as directional; local thermal state, memory pressure, and host load can move the raw numbers.

| mode | prompt | tok/s | accepted |
|---|---|---:|---:|
| no spec | numbers | `20.56` | - |
| MTP nmax=1 | numbers | `23.29` | `31/31` |
| MTP nmax=1, blend/bias | numbers | `26.02` | `31/31` |
| no spec | speculative_explain | `21.02` | - |
| MTP nmax=1 | speculative_explain | `21.90` | `28/35` |
| MTP nmax=1, blend/bias | speculative_explain | `23.48` | `29/33` |

`blend/bias` here means `--spec-logit-blend 2.0 --spec-accept-bias 4.0`. Treat it as heuristic and evaluate output quality yourself.

That means this is usable as an experimental MiMo MTP path, but it is not yet a broad SGLang-like speed path.
