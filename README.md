# llama.go

SmolLM2-135M inference in pure Go, compiled to WebAssembly.

<https://bquast.github.io/llama.go>

## description
**llama.go** is a minimalist, dependency-free LLM inference engine designed to run SmolLM2-135M-Instruct directly in the browser. Inspired by [llama2.c](https://github.com/karpathy/llama2.c) and [llama.cpp](https://github.com/ggerganov/llama.cpp), this project implements the transformer architecture from scratch in Go without relying on external machine learning libraries.

The engine parses GGUF files (specifically the Q8_0 or Q4_K_M variants) which are self-contained, embedding the BPE tokenizer vocabulary and merge rules alongside the quantized weights. It is built for the WebAssembly (WASM) target, providing a private, local, and lightweight AI chat experience served via vanilla HTML, JS, and CSS.

### key features
- **Pure Go:** No CGO or external ML frameworks.
- **Zero-Footprint:** Self-contained GGUF parsing and BPE tokenization.
- **Minimalist UI:** Vanilla frontend with browser-native caching via the Cache API.
- **Functional Style:** Linear, readable code prioritizing clarity and ease of debugging.

## todo
- [ ] **Wasm SIMD Optimization:** Implement 128-bit SIMD instructions for `matVec` and `dot` kernels to achieve 3-5x performance gains.
- [ ] **Native Tokenizer:** Translate the `llama2.c` tokenizer logic directly into the Go source to remove the `regexp` dependency and reduce binary size.
- [ ] **JS-Side Templating:** Move ChatML string formatting (e.g., `<|im_start|>`) to the JavaScript layer to treat the Go engine as a pure token-in/token-out machine.
- [ ] **Performance Benchmarking:** Add telemetry to track tokens-per-second (TPS) and memory overhead across different browsers.
- [ ] **Extended GGUF Support:** Improve the parser to handle a wider variety of quantization types and architectural variations.
