[toc]

# 大模型基础知识

# 推理框架

vLLM Text Generation Inference Faster Transformer TRT LLM

ONNX

## LLM推理优化

FlashAttention Continuous Batching KV Cache Flash Decoding FlashDecoding++

Decoder-only inference

Speculative decoding Speculative decoding: small off-the-shelf model Speculative decoding: n-grams Speculative decoding: Medusa

## LLM量化

量化基础

量化感知训练

Dynamic post-training quantization with PyTorch

训练后量化

ZeroQuant

bitsandbytes

SmoothQuant

Group-wise Precision Tuning Quantization (GPTQ

Activation-aware Weight Quantization (AWQ)

Half-Quadratic Quantization (HQQ)


## LLM推理优化技术-概述
https://github.com/liguodongiot/llm-action

大模型推理优化技术-KV Cache
大模型推理服务调度优化技术-Continuous batching
大模型低显存推理优化-Offload技术
大模型推理优化技术-KV Cache量化
大模型推理优化技术-张量并行
大模型推理服务调度优化技术-Chunked Prefill
大模型推理优化技术-KV Cache优化方法综述
大模型吞吐优化技术-多LoRA推理服务
大模型推理服务调度优化技术-公平性调度
大模型访存优化技术-FlashAttention
大模型显存优化技术-PagedAttention
大模型解码优化-Speculative Decoding及其变体
大模型推理优化-结构化文本生成
Flash Decoding
FlashDecoding++