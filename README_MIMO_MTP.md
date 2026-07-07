# MiMo-V2.5 MTP Speculative Decoding

Same-GGUF MTP speculative decoding support for MiMo-V2.5 in `llama.cpp`.

This fork loads the normal MiMo-V2.5 target model and uses the MiMo `nextn` / MTP tensors from the same GGUF through llama.cpp's `draft-mtp` path. No separate draft model file is required.

## Current State

This branch is based on current upstream `llama.cpp` with a small MiMo-specific patch on top. The goal is to stay close to upstream's current MTP implementation rather than carry the older native-MTP experiment.

Tested locally:

- Apple M3 Max, Metal, 128 GB unified memory.
- Linux CUDA rig with RTX 3090 + Tesla P40 + DDR5 system memory.
- MiMo-V2.5 IQ3_S GGUF with appended MTP tensors.
- AesSedai-style MiMo-V2.5 IQ3_S GGUF with appended MTP tensors.
- MiMo-V2.5 IQ2_XXS / Q8-infra GGUF with BF16 MTP tensors.
- `llama-server --spec-type draft-mtp --spec-draft-n-max 1`.

Works:

- MiMo-V2.5 target inference with appended MTP / `nextn` tensors.
- MiMo target graph skips appended MTP layers during normal inference.
- `draft-mtp` uses the MiMo MTP block from the same GGUF.
- Conservative `nmax=1` serving.
- CUDA and Metal smoke tests.

Not the focus of this clean branch:

- Experimental native-MTP scheduling from the older worktree.
- `nmax=2+` as a speed path.
- MTP3-style external draft models.
- Custom acceptance/logit-blending heuristics.

## Published GGUFs

The original tested IQ3_S same-GGUF build is available at [tnhnyzc/MiMO-V2.5-MTP-GGUF](https://huggingface.co/tnhnyzc/MiMO-V2.5-MTP-GGUF).

The current homelab quant being tested is an IQ2_XXS / Q8-infra MiMo-V2.5 GGUF with BF16 MTP tensors. It is intended for this fork until equivalent MiMo MTP support is available in upstream llama.cpp.

Quant recipes, briefly:

- IQ3_S build: follows AesSedai's recipe; dense/infra `Q6_K`, expert gate/up/down `IQ2_S`, MTP tensors appended.
- IQ2_XXS / Q8-infra build: dense/infra `Q8_0`, expert gate/up/down `IQ2_XXS`, `nextn.eh_proj` `BF16`, norms/biases left `F32`.

## Quick Start

Build with your backend as usual. For CUDA:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON
cmake --build build-cuda --target llama-server -j
```

Run the conservative MTP mode:

```bash
./build-cuda/bin/llama-server \
  --model /path/to/MiMo-V2.5-MTP.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 1 \
  -np 1 \
  -ngl 99 \
  -fa on
```

Add your normal sampling, cache, and placement arguments as needed. The tested serving path is `llama-server`.

## Runtime Notes

Required:

- `general.architecture = mimo2`.
- GGUF includes MiMo MTP / `nextn` tensors.
- `mimo2.nextn_predict_layers > 0`.
- `--spec-type draft-mtp`.
- `--spec-draft-n-max 1` is the recommended starting point.

Recommended:

- Start with `--spec-draft-n-max 1`.
- Use `-np 1` while validating behavior.
- Use `--flash-attn on` where supported.
- Keep MTP projection/norm/head tensors high precision when making very small target quants.
- Benchmark with your real serving pattern, not only short fresh prompts.

## Example CUDA Placement

One tested RTX 3090 + Tesla P40 setup used a mixed GPU/CPU placement for a large MiMo-V2.5 IQ2_XXS / Q8-infra quant:

```bash
--ctx-size 32768
-ctk q8_0 -ctv q8_0
-ngl 99
-b 2048
-ub 1024
-fa on
--tensor-split "0.99,0.01"
--fit off
-ot "token_embd.weight=CUDA0,output_norm.weight=CUDA0,output.weight=CUDA0,blk\.0\..*=CUDA0,blk\.(48|49|50)\..*=CUDA0,blk\.[1-9]\.ffn_.*_exps\.weight=CUDA0,blk\.(1[0-9]|2[0-3])\.ffn_.*_exps\.weight=CUDA1,blk\.(2[4-9]|3[0-9]|4[0-7])\.ffn_.*_exps\.weight=CPU"
--spec-type draft-mtp
--spec-draft-n-max 1
```

This is hardware-specific. Treat it as a useful reference point, not a universal recommendation.

## Expected Performance

MTP speedup is workload-dependent. Short, fresh synthetic prompts can understate the benefit because MTP has overhead and the server/cache state is not representative of real chat serving.

On an RTX 3090 + Tesla P40 + DDR5 rig with the IQ2_XXS / Q8-infra + BF16 MTP GGUF above:

| mode | workload | observed generation speed |
|---|---|---:|
| no spec | real cached chat serving | about `18.3 t/s` average |
| MTP nmax=1 | real cached chat serving | about `22.6 t/s` average, with runs near `24-25 t/s` |
| no spec | short fresh direct `/completion` prompts | about `20.1 t/s` |
| MTP nmax=1 | short fresh direct `/completion` prompts | about `20.2 t/s` |

In this setup, the real cached chat workload showed roughly a 20-25% generation-speed uplift with MTP enabled. The short prompt benchmark was nearly neutral. Hardware placement, cache reuse, prompt shape, generation length, and host load all matter.

On Apple M3 Max / Metal with IQ3_S GGUFs, the branch built and smoke-tested successfully with both a local MiMo-V2.5 IQ3_S quant and an AesSedai-style IQ3_S quant. A short single-turn smoke with the local IQ3_S quant and `--spec-type draft-mtp --spec-draft-n-max 1` generated normally at about `34 t/s`. Longer short-prompt comparisons were prompt-dependent and did not show the same clear uplift as the cached CUDA serving workload.

## Legacy Branch

The previous public fork state is preserved as `legacy-native-mtp-2026-07-07`. That branch contains the older experimental native-MTP work and documentation. The current `main` branch is intentionally cleaner: it is current upstream `llama.cpp` plus MiMo support for upstream's `draft-mtp` path.
