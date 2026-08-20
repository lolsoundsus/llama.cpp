# Unsloth's llama.cpp + Anbeeld's beellama.cpp
Fully edited by Big Pickle in OpenCode.
LLM inference in C/C++ with Variance-Normalized KV-Cache Quantization.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This is an independent build of [llama.cpp](https://github.com/ggml-org/llama.cpp) with [KVarN](https://github.com/Anbeeld/beellama.cpp) (Variance-normalized KV-cache quantization) merged in, providing higher precision at similar memory costs compared to standard KV-cache quantization.

## What's Different

This fork combines the latest upstream `llama.cpp` with KVarN from [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) and optimizations from [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp).

### KVarN: Variance-Normalized KV-Cache Quantization

KVarN provides **higher precision KV-cache quantization at similar memory costs** by normalizing against variance rather than using uniform quantization.

**Features:**
- Independent K and V bit widths: `kvarn2`, `kvarn3`, `kvarn4`, `kvarn5`, `kvarn6`, and `kvarn8`
- Set via `--cache-type-k` and `--cache-type-v` flags
- Native CUDA FlashAttention with MMA support for KVarN
- Vulkan KVarN compute shaders
- Portable KVarN attention for pre-Turing CUDA and AMD GPUs
- Rotated-domain attention for KVarN decode
- Prompt cache and checkpoint support for KVarN

**Usage:**
```sh
# Use kvarn4 for both K and V caches
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF --cache-type-k kvarn4 --cache-type-v kvarn4

# Different bit widths for K and V
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF --cache-type-k kvarn8 --cache-type-v kvarn4

# KVarN8 for key, standard quantization for value
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF --cache-type-k kvarn8 --cache-type-v q8_0
```

### Cache Type Reference

| Cache Type | Description |
|---|---|
| `kvarn2` | 2-bit KVarN quantization |
| `kvarn3` | 3-bit KVarN quantization |
| `kvarn4` | 4-bit KVarN quantization |
| `kvarn5` | 5-bit KVarN quantization |
| `kvarn6` | 6-bit KVarN quantization |
| `kvarn8` | 8-bit KVarN quantization |
| `f16` | Float16 (no quantization) |
| `q8_0` | Standard 8-bit quantization |
| `q4_0` | Standard 4-bit quantization |

## Quick Start

Build from source:

```sh
git clone https://github.com/lolsoundsus/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON  # Enable CUDA support
cmake --build build --config Release -j$(nproc)
```

Or run with Docker:

```sh
docker run -it --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  --host 0.0.0.0 --port 8080 \
  -hf ggml-org/Qwen3.5-0.8B-GGUF \
  --cache-type-k kvarn4 --cache-type-v kvarn4
```

## Supported Backends

| Backend | Target devices |
|---|---|
| [CUDA](docs/build.md#cuda) | NVIDIA GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| CPU | All (x86, ARM, RISC-V) |

## Building

See the [build guide](docs/build.md) for detailed instructions.

### CUDA Build

```sh
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

### CPU-Only Build

```sh
cmake -B build
cmake --build build --config Release -j$(nproc)
```

### With KVarN Support

KVarN is enabled by default. Use `--cache-type-k` and `--cache-type-v` at runtime to activate.

## Tested Hardware

Currently tested on **Linux x86_64 with CUDA**:
- AMD Ryzen 7 7745HX
- 2x NVIDIA RTX 4060 Ti 16GB

Other platforms (Windows, macOS, AMD HIP, Vulkan) are not yet tested. YMMV.

## Documentation

- [Build guide](docs/build.md)
- [Server API](tools/server/README.md)
- [CLI usage](tools/cli/README.md)
- [Multi-GPU](docs/multi-gpu.md)
- [Docker](docs/docker.md)

## Credits

This project combines work from multiple sources:

- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** - The upstream llama.cpp project
- **[Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp)** - KVarN (Variance-normalized KV-cache quantization) implementation
- **[unslothai/llama.cpp](https://github.com/unslothai/llama.cpp)** - Unsloth's optimizations and prebuilt infrastructure

### Third-Party Libraries

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - HTTP server - MIT
- [nothings/stb](https://github.com/nothings/stb) - Image decoder - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - JSON library - MIT
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Audio decoder - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Process launching - Public domain

## License

MIT - same as upstream [llama.cpp](https://github.com/ggml-org/llama.cpp).
