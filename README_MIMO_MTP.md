# MiMo-V2.5 MTP Speculative Decoding

Experimental same-GGUF MTP speculative decoding support for MiMo-V2.5 in `llama.cpp`.

This fork loads the normal MiMo-V2.5 target model and uses MiMo's appended `nextn` / MTP tensors from the same GGUF through llama.cpp's `draft-mtp` path. No separate draft model file is needed.

## Current State

This branch is based on upstream `llama.cpp` as of July 2026, with a small MiMo-specific patch on top.

Tested locally:

- Apple M3 Max, Metal, 128 GB unified memory.
- Linux CUDA with RTX 3090 + Tesla P40 + DDR5 system memory.
- MiMo-V2.5 IQ3_S GGUF with appended BF16 MTP tensors.
- MiMo-V2.5 IQ2_XXS / Q8-infra GGUF with BF16 MTP tensors.
- `llama-server --spec-type draft-mtp --spec-draft-n-max 1`.

The tested path is conservative `nmax=1` serving. Higher draft lengths may work, but are not the recommended starting point.

## Published GGUFs

Published same-GGUF MTP builds are available at [tnhnyzc/MiMO-V2.5-MTP-GGUF](https://huggingface.co/tnhnyzc/MiMO-V2.5-MTP-GGUF).

Quant recipes, briefly:

- IQ3_S build: follows AesSedai's recipe; dense/infra `Q6_K`, expert gate/up/down `IQ2_S` / `IQ2_S` / `IQ3_S`, MTP tensors appended as `BF16`.
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

Add your normal sampling, cache, and placement arguments as needed.

## Runtime Notes

Required:

- `general.architecture = mimo2`.
- GGUF includes MiMo MTP / `nextn` tensors.
- `mimo2.nextn_predict_layers > 0`.
- `--spec-type draft-mtp`.

Recommended:

- Start with `--spec-draft-n-max 1`.
- Use `-np 1` while validating behavior.
- Use `--flash-attn on` where supported.
- Keep MTP projection/norm/head tensors high precision when making very small target quants.
- Benchmark with your real serving pattern, not only short fresh prompts.

## Expected Performance

MTP speedup is workload-dependent. Short, fresh synthetic prompts can understate the benefit because MTP has overhead and the server/cache state is not representative of real chat serving.

| setup | mode | workload | observed generation speed |
|---|---|---|---:|
| RTX 3090 + Tesla P40 + DDR5, IQ2_XXS / Q8-infra + BF16 MTP | no spec | real cached chat serving | about `18.3 t/s` average |
| RTX 3090 + Tesla P40 + DDR5, IQ2_XXS / Q8-infra + BF16 MTP | MTP nmax=1 | real cached chat serving | about `22.6 t/s` average, with runs near `24-25 t/s` |
| RTX 3090 + Tesla P40 + DDR5, IQ2_XXS / Q8-infra + BF16 MTP | no spec | short fresh direct `/completion` prompts | about `20.1 t/s` |
| RTX 3090 + Tesla P40 + DDR5, IQ2_XXS / Q8-infra + BF16 MTP | MTP nmax=1 | short fresh direct `/completion` prompts | about `20.2 t/s` |

On the tested CUDA setup, real cached chat serving showed roughly a 20-25% generation-speed uplift with MTP enabled. Short direct prompt tests were mostly neutral.

Apple M3 Max / Metal was also benchmarked with the IQ3_S GGUF with BF16 MTP tensors:

| setup | mode | workload | observed generation speed |
|---|---|---|---:|
| Apple M3 Max / Metal, IQ3_S + BF16 MTP | no spec | 3 short direct `/completion` prompts | about `31.4 t/s` average |
| Apple M3 Max / Metal, IQ3_S + BF16 MTP | MTP nmax=1 | same 3 short direct `/completion` prompts | about `29.7 t/s` average |

The Metal MTP run accepted draft tokens on those prompts, with observed acceptance around `78-86%`, but this small short-prompt benchmark did not show a speedup.

Hardware placement, cache reuse, prompt shape, generation length, backend, and host load all matter.

## Legacy Branch

The previous public fork state is preserved as `legacy-native-mtp-2026-07-07` for reference. The current `main` branch is a cleaner upstream-based implementation using llama.cpp's `draft-mtp` path.
