# Rust Ecosystem for AI & LLMs

Rust’s strengths in performance, safety, and concurrency are gaining ground in the demanding realm of AI and Large Language Models (LLMs). Once a niche language in this space, Rust now supports a vibrant and growing ecosystem of machine learning tools—from lean inference engines to robust vector database clients.

However, this rapid expansion can make it difficult to pinpoint the right crate for your needs—whether you're deploying models on edge devices, fine-tuning transformers, or managing prompt flows via API.

This article offers a structured reference to guide your selection. We've categorized key Rust libraries across functional domains, including:

## Inference Engines: 
[Candle](https://github.com/huggingface/candle) – A minimalist ML framework supporting fast CPU/GPU inference with models like Transformers and Whisper. Backed by Hugging Face.

## LLM API Clients:

[Ollama-rs](https://github.com/pepperoni21/ollama-rs) – A Rust client for Ollama, enabling Rust applications to interact with locally served LLMs.

## Vector Stores:
[Qdrant-Client](https://crates.io/crates/qdrant-client) – A robust client for the Qdrant vector database, used to store and query embeddings efficiently.

Whether you're evaluating Rust as an alternative to Python-based stacks or seeking low-latency solutions for production, this guide highlights the tools that can help you unlock Rust’s full potential for AI.
## Inference Engines and Runtime Libraries

| **Crate/Repo Name** | **Category** | **Description** | **Status** |
| :-------------------------------------------------------- | :--------------- | :------------------------------------------------------------------------------------------------------------- | :--------------------- |
| **[Candle](https://github.com/huggingface/candle)** | Inference Engine | Minimalist Rust ML framework with fast CPU/GPU inference (supports Transformers, Whisper, etc.)                    | Active (Hugging Face)  |
| **[Mistral.rs](https://github.com/EricLBuehler/mistral.rs)** | Inference Engine | Rust implementation for running quantized LLaMA/Mistral models efficiently (Apple Silicon, CPU, CUDA)            | Active                 |
| **[Ratchet](https://github.com/huggingface/ratchet)** | Inference Engine | Cross-platform WebGPU-based ML runtime for browser & native (focus on LLM inference on GPU via WebGPU)         | Active (in dev)        |
| **[Llama-rs](https://github.com/rustformers/llama-rs)** | Inference Engine | Rust implementation of LLaMA (GGML-based backend) for local inference (Rustformers project)                      | *Archived* (unmaintained) |
| **[Rustformers LLM](https://github.com/rustformers/llm)** | Inference Engine | Rust ecosystem around GGML for multiple LLMs (BLOOM, GPT-NeoX, etc.) – now deprecated                            | *Archived* (deprecated) |
| **[Rust-BERT](https://github.com/guillaume-be/rust-bert)** | Inference Engine | Ready-to-use NLP pipelines (e.g. QA, summarization) leveraging transformer models in Rust (uses Torch backend) | Active                 |
| **[Oxidized-Transformers](https://github.com/oxidized-transformers/oxidized-transformers)** | Inference Engine | Pure Rust implementation of Hugging Face transformers (experimental support for GPT/BERT models)             | Experimental           |
| **[CallM](https://crates.io/crates/callm)** | Inference Engine | High-level API to run local LLMs (LLaMA2, Mistral, etc.) on CPUs/GPUs using Candle backend                      | Active                 |
| **[llm_client](https://crates.io/crates/llm_client)** | Inference Engine | High-level Rust wrapper for `llama.cpp` to run GPT-style models locally (streaming support, simple API)        | Active                 |
| **[llama_cpp](https://crates.io/crates/llama_cpp)** | Inference Engine | Safe, high-level Rust bindings to `llama.cpp` (LLaMA model C++ library)                                        | Active                 |
| **[llama-cpp-2](https://crates.io/crates/llama-cpp-2)** | Inference Engine | Low-level Rust binding closely mirroring `llama.cpp` C API (thin wrapper)                                      | Active                 |
| **[DramaLlama](https://crates.io/crates/drama_llama)** | Inference Engine | “drama_llama” – Ergonomic, high-level Rust API over llama.cpp for chat completions                             | Active                 |
| **[Kalos](https://crates.io/crates/kalosm)** | Inference Engine | Candle-based simple interface for multimodal models (language, audio, image)                                   | Active                 |
| **[LlamaEdge](https://github.com/LlamaEdge/LlamaEdge)** | Inference Engine | Rust+Wasm runtime to deploy fine-tuned LLMs on edge devices (GGUF format models via Wasm)                      | Active                 |
| **[Edge Transformers](https://crates.io/crates/edge-transformers)** | Inference Engine | Rust ONNX Runtime wrapper implementing Hugging Face “Optimum” transformer pipelines (for edge/native inference) | Active                 |
| **[Tract](https://github.com/sonos/tract)** | Inference Engine | Self-contained neural network inference in Rust (supports ONNX, TensorFlow models)                             | Active                 |
| **[Ort](https://github.com/pykeio/ort)** | Inference Engine | Rust bindings for ONNX Runtime (Microsoft) to run ONNX models on CPU/GPU (supersedes older `onnxruntime-rs`)   | Active                 |
| **[WONNX](https://crates.io/crates/wonnx)** | Inference Engine | 100% Rust ONNX inference runtime using wgpu (WebGPU) – run models in browser or natively                       | Active                 |
| **[TensorFlow Rust](https://github.com/tensorflow/rust)** | Inference Engine | Official Rust bindings for TensorFlow C API (run pre-trained TF models in Rust)                                | Low-Activity           |
| **[Apache MXNet Rust](https://github.com/apache/incubator-mxnet/tree/master/rust)** | Inference Engine | Rust bindings for MXNet deep learning framework (supports model inference/training via MXNet C++ backend)      | *Archived* |

## Tokenizers and Text Preprocessing

| **Crate/Repo Name** | **Category** | **Description** | **Status** |
| :--------------------------------------------------------- | :------------------ | :------------------------------------------------------------------------------------------------------------- | :------------------------- |
| **[Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)** | Tokenizer Library   | Fast BPE/WordPiece tokenizers (Rust + HF’s `tokenizers` library) for Transformers (used for GPT-2, BERT, etc.) | Active                     |
| **[tiktoken](https://github.com/openai/tiktoken)** | Tokenizer Library   | OpenAI’s BPE tokenizer for GPT-3/4 models (Rust-backed via Python) – enables tokenizing text exactly as OpenAI models do | Active                     |
| **[tiktoken-rs](https://github.com/anysphere/tiktoken-rs)** | Tokenizer Library   | Unofficial pure-Rust port of OpenAI’s tiktoken library (experimental, supports GPT-4 encodings)                | Experimental               |
| **[rust-tokenizers](https://crates.io/crates/rust-tokenizers)** | Tokenizer Library   | High-performance Rust tokenizer implementations for popular models (BERT WordPiece, SentencePiece, etc.)         | Active                     |
| **[SentencePiece (Rust)](https://crates.io/crates/sentencepiece)** | Tokenizer Library   | Rust binding to Google’s SentencePiece tokenizer (unigram language model segmentation)                         | Active                     |
| **[Lindera](https://github.com/lindera-morphology/lindera)** | Text Processing     | Japanese tokenizer (morphological analyzer) in Rust (equivalent to MeCab) for tokenizing Japanese text         | Active                     |
| **[Sudachi.rs](https://github.com/WorksApplications/sudachi.rs)** | Text Processing     | Japanese tokenizer binding (Sudachi dictionary) in Rust for comprehensive Japanese tokenization              | Active                     |
| **[whatlang-rs](https://github.com/greyblake/whatlang-rs)** | Text Detection      | Language identification library for text (detects language from input text)                                    | Active                     |
| **[lingua-rs](https://github.com/pemistahl/lingua-rs)** | Text Detection      | Highly accurate natural language detection (detects 75+ languages) ported to Rust (based on Lingua)            | Active                     |
| **[NLPRule](https://github.com/bminixhofer/nlprule)** | Text Processing     | Rule-based text preprocessing (for grammar and spelling correction) in Rust (fast and low-resource)            | Active                     |
| **[nnsplit](https://github.com/bminixhofer/nnsplit)** | Text Segmentation   | Neural sentence segmentation – splits text into sentences or semantic units (useful for chunking long texts)   | Active                     |
| **[text-splitter](https://github.com/benbrandt/text-splitter)** | Text Segmentation   | Utility to split text into chunks of a target length (for prompt chunking in LLM apps)                       | Active                     |
| **[Tokenizations](https://github.com/robinstanley/tokenizations)** | Token Alignment     | Library to align tokenizer output with original text (useful for mapping token indices to substrings)        | Active                     |
| **[Tokengrams](https://github.com/EleutherAI/tokengrams)** | Text Processing     | Tool from EleutherAI for computing and storing token n-grams from large text corpora (for dataset analysis)    | Experimental               |
| **[SPM Precompiled](https://crates.io/crates/spm_precompiled)** | Tokenizer Utility   | A specialized crate for parsing SentencePiece models (SPM) without requiring the C++ library at runtime        | Active                     |
| **[rust-stop-words](https://crates.io/crates/rust-stop-words)** | Text Processing     | Lists of common stopwords in various languages (for text cleaning in NLP pipelines)                          | Active                     |
| **[Vader Sentiment (Rust)](https://github.com/ckw017/vader-sentiment-rust)** | Text Analysis       | Rust port of VADER sentiment analysis (rule-based sentiment scoring for social media text)                   | Active                     |
| **[fastText (Rust binding)](https://crates.io/crates/fasttext)** | Text Embeddings     | Rust binding to Facebook’s fastText library (for word embeddings and text classification)                      | Maintained (minor updates) |

## Model Training & Fine-Tuning Frameworks

| **Crate/Repo Name** | **Category** | **Description** | **Status** |
| :--------------------------------------------------------- | :----------------------- | :------------------------------------------------------------------------------------------------------------- | :------------- |
| **[Burn](https://github.com/burn-rs/burn)** | Training Framework       | Comprehensive Rust deep learning framework (autodiff, modules, training APIs) with multiple backends             | Active         |
| **[DFDX](https://github.com/coreylowman/dfdx)** | Training Framework       | Pure Rust deep learning library (“Diffusion” framework) – dynamic shape tensors, autograd, GPU support         | Active         |
| **[tch-rs](https://github.com/LaurentMazare/tch-rs)** | Training Framework       | Rust bindings for PyTorch (LibTorch C++ API) – enables training and inference with Torch models in Rust          | Active         |
| **[Autodiff](https://crates.io/crates/autodiff)** | Training Utility         | General automatic differentiation library in Rust (supports forward and reverse mode AD)                       | Active         |
| **[Luminal](https://github.com/jafioti/luminal)** | Training/Infer Framework | Experimental Rust ML library using compile-time optimized GPU kernels (targets high performance on CUDA/Metal) | Experimental   |
| **[Juice](https://github.com/spearow/juice)** | Training Framework       | Early Rust DL framework (AutumnAI project) featuring GPU support via Coaster – now superseded by newer libs    | *Archived* |
| **[Dora](https://github.com/dora-rs/dora)** | MLOps / Pipeline         | Distributed orchestration for ML experiments in Rust (pipelines, model serving) – useful for chaining LLM modules | Active         |
| **[TensorFlow Rust](https://github.com/tensorflow/rust)** | Training Framework       | Rust bindings for TensorFlow (train or run TF models). Supports Graph and Eager execution via TF C API.        | Low-Activity   |
| **[Apache MXNet Rust](https://github.com/apache/mxnet/tree/master/rust)** | Training Framework       | Rust API for MXNet (deep learning framework) supporting model training/inference with NDArray, Symbol APIs   | *Archived* |

## LLM Provider Clients and SDKs

| **Crate/Repo Name** | **Category** | **Description** | **Status** |
| :------------------------------------------------------------------- | :----------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :------------- |
| **[OpenAI (Rust SDK)](https://crates.io/crates/openai)** | LLM API Client     | Unofficial Rust library for OpenAI API (completions, chat, etc.) – asynchronous support, mirrors OpenAI endpoints                               | Active         |
| **[openai-api-rs](https://crates.io/crates/openai-api-rs)** | LLM API Client     | Alternative OpenAI API client with similar features (community-maintained)                                                                      | Active         |
| **[async-openai](https://github.com/64bit/async-openai)** | LLM API Client     | Fully async OpenAI client in Rust (provides both low-level and high-level APIs for OpenAI services)                                           | Active         |
| **[openai-safe](https://crates.io/crates/openai-safe)** | LLM API Client     | Type-safe OpenAI API bindings (strongly typed requests/responses to reduce errors)                                                            | Active         |
| **[Azure OpenAI (az-openai-rs)](https://crates.io/crates/az-openai-rs)** | LLM API Client     | Client for Azure’s OpenAI Service (compatible with OpenAI API, using Azure endpoints/keys)                                                    | Active         |
| **[Anthropic SDK (anthropic-rs)](https://crates.io/crates/anthropic)** | LLM API Client     | Rust client for Anthropic’s Claude API (streaming completions, etc.)                                                                          | Active (WIP)   |
| **[Misanthropy](https://github.com/cortesi/misanthropy)** | LLM API Client     | Another Rust binding for Anthropic’s API (easy access to Claude models)                                                                       | Experimental   |
| **[Cohere-Rust](https://crates.io/crates/cohere-rust)** | LLM API Client     | Rust SDK for Cohere’s NLP API (generate, embed, classify via Cohere’s large models)                                                           | Active         |
| **[Hugging Face Hub (hf-hub)](https://crates.io/crates/hf-hub)** | Model Hub Client   | Rust client to download models/datasets from HuggingFace Hub (mirrors Python `huggingface_hub`)                                                 | Active         |
| **[HuggingFace Inference](https://crates.io/crates/huggingface_inference_rs)** | LLM API Client     | Client for HuggingFace Inference Endpoint API (call hosted models for completions)                                                            | Active         |
| **[Ollama-rs](https://github.com/pepperoni21/ollama-rs)** | LLM API Client     | Rust client for Ollama (which serves local models via an API) – lets Rust apps use Ollama’s LLMs                                              | Active         |
| **[AWS Bedrock SDK](https://crates.io/crates/aws-sdk-bedrock)** | LLM API Client     | Rust SDK integration for AWS Bedrock (invoke foundation models like Anthropic Claude, AI21, etc. via AWS)                                     | Active         |
| **[Allms](https://crates.io/crates/allms)** | Multi-API Client   | Unified client for many LLM APIs (OpenAI, Anthropic, Mistral, etc.), sharing common interface                                                 | Active         |
| **[llm (RLLM)](https://crates.io/crates/llm)** | Multi-API Client   | “RLLM” – Single API to multiple backends (OpenAI, Anthropic, Ollama, xAI, etc.) with unified chat/completion traits                           | Active         |
| **[llmclient](https://crates.io/crates/llmclient)** | Multi-API Client   | Another unified client for various LLM providers (Gemini, OpenAI, Anthropic, etc.)                                                            | Active         |
| **[rust-genai](https://github.com/jeremychone/rust-genai)** | Multi-API Client   | “genAI” multi-provider SDK (OpenAI, Anthropic, xAI, Groq, Ollama, etc.) – single interface to many AI services                               | Experimental   |
| **[tysm](https://github.com/not-pizza/tysm)** | OpenAI Client      | High-level OpenAI client focusing on ease-of-use (“batteries-included”; supports batching and embeddings)                                       | Active         |
| **[ChatGPT Rust](https://crates.io/crates/chatgpt)** | OpenAI Client      | Specialized wrapper for ChatGPT (manages conversations, message history, function calling support)                                            | Active         |

## Prompt Engineering & Orchestration Frameworks

| **Crate/Repo Name** | **Category** | **Description** | **Status** |
| :---------------------------------------------------------------------- | :----------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- | :------------- |
| **[llm-chain](https://github.com/sobelio/llm-chain)** | Prompt Orchestration     | “Ultimate toolbox” for building LLM apps in Rust – chain prompts, manage state, includes utils (embeddings, etc.)                              | Active         |
| **[LangChain-rust](https://github.com/Abraxas-365/langchain-rust)** | Prompt Orchestration     | Rust port of LangChain concepts – composable prompts, chains, agents, memory (inspired by LangChain.py)                                       | Active         |
| **[Anchor-Chain](https://crates.io/crates/anchor-chain)** | Prompt Orchestration     | Statically-typed, async framework for LLM workflows – define agents/tools in Rust (YAML or code)                                               | Active         |
| **[Rig](https://github.com/0xPlaygrounds/rig)** | LLM Applications         | Modular framework for building LLM-powered apps (unified interface, agent abstractions, Retrieval-Augmented Generation support)                  | Active         |
| **[Outlines](https://github.com/dottxt-ai/outlines-core)** | Structured Generation    | Library for guiding LLM output with structural templates (ensures output conforms to a schema)                                                 | Active         |
| **[AICI](https://github.com/microsoft/aici)** | Prompt Programming       | Microsoft’s “AICI: Prompts as WASM programs” – experimental framework treating prompts as code (WASM-based)                                    | Experimental   |
| **[Swiftide](https://github.com/bosun-ai/swiftide)** | RAG/Agents Framework     | Library for building fast RAG pipelines and agents in Rust (ingest data, vector index, prompt injection)                                       | Active         |
| **[Kwaak](https://github.com/bosun-ai/kwaak)** | Autonomous Agents        | Bosun’s open-source app for running multiple AI coding agents in parallel on a codebase (uses Swiftide under the hood)                         | Active         |
| **[Nerve](https://github.com/evilsocket/nerve)** | Autonomous Agents        | YAML-based agent creation kit – define tools and flows in YAML, run stateful multi-step LLM agents (CLI utility)                             | Active         |
| **[CrustAGI](https://github.com/lukaesch/CrustAGI)** | Autonomous Agents        | “BabyAGI” port to Rust – an AI task management agent that uses GPT to create/execute tasks iteratively                                       | Experimental   |
| **[SmartGPT (rust)](https://crates.io/crates/smartgpt)** | Autonomous Agents        | Framework for modular LLM agents – separates LLMs, tools, memory, etc. for building AutoGPT-like systems                                     | Experimental   |
| **[AutoGPT-rs](https://crates.io/crates/autogpt)** | Autonomous Agents        | Rust implementation of an Auto-GPT style agent framework (automate tasks with GPT-4, minimal human input)                                    | Experimental   |
| **[llm-chain-hnsw](https://crates.io/crates/llm-chain-hnsw)** | Retrieval Utility        | Integration of `llm-chain` with HNSW vector search – enables semantic retrieval of context for prompts                                         | Active         |
| **[RAG-Toolchain](https://crates.io/crates/rag-toolchain)** | RAG Pipeline             | Rust toolkit for Retrieval-Augmented Generation workflows (document ingestion, indexing, query processing for LLMs)                          | Active (early) |

## Embeddings, Vector Stores, and Other Utilities

| **Crate/Repo Name** | **Category** | **Description** | **Status** |
| :---------------------------------------------------------------- | :--------------------- | :------------------------------------------------------------------------------------------------------------- | :--------------------- |
| **[SafeTensors](https://crates.io/crates/safetensors)** | Model Serialization    | Rust implementation of safetensors – safe, zero-copy format for storing model weights (used instead of PyTorch pickle) | Active                 |
| **[Rust-SBERT](https://github.com/cpcdoy/rust-sbert)** | Embeddings             | Rust port of SentenceTransformers – generate sentence embeddings with pre-trained models (e.g. MiniLM)           | Active                 |
| **[FAISS-RS](https://crates.io/crates/faiss)** | Vector Search          | Rust bindings for Facebook’s FAISS library (high-speed similarity search over embeddings)                        | Active                 |
| **[HNSW-RS](https://crates.io/crates/hnsw_rs)** | Vector Search          | Pure Rust implementation of HNSW algorithm for ANN search (approx. nearest neighbors in vector space)            | Active                 |
| **[Qdrant-Client](https://crates.io/crates/qdrant-client)** | Vector DB Client       | Rust client for Qdrant vector database (for storing and querying embedding vectors)                              | Active                 |
| **[LanceDB](https://github.com/lancedb/lancedb)** | Vector Store           | Embeddable vector database written in Rust (for similarity search, supports disk-based indices)                  | Active                 |
| **[PGVectorScale](https://github.com/timescale/pgvectorscale)** | Vector DB Utility      | Tooling to scale Postgres + PGVector for AI apps (TimescaleDB project for sharding vector search)                | Experimental           |
| **[Ndarray](https://crates.io/crates/ndarray)** | Tensor Utility         | N-dimensional array crate for Rust (used for matrix/tensor ops, common in ML computations)                       | Active                 |
| **[Half](https://crates.io/crates/half)** | Numeric Utility        | Crate providing f16 (half-precision float) type – useful for 16-bit model weights and quantization             | Active                 |
| **[Tabby](https://github.com/TabbyML/tabby)** | LLM Application        | Self-hosted AI coding assistant (server that runs code completion models, OpenAI-compatible API)                 | Active                 |
| **[Text-Gen Inference](https://github.com/huggingface/text-generation-inference)** | LLM Serving            | High-performance text-generation server in Rust (by HuggingFace) for hosting transformers with gRPC/HTTP APIs  | Active                 |
| **[Ungoliant](https://github.com/oscar-project/ungoliant)** | Data Pipeline          | Pipeline for cleaning and preprocessing web text (OSCAR corpus) in Rust – useful for creating LLM training datasets | Active (data prep)   |