# Recorded environment

- Original method run: 2026-07-19 UTC
- Ordinary-SFT controls and package audit: 2026-07-22 to 2026-07-24 UTC
- OS: Ubuntu/glibc 2.35, Linux 5.4.0
- Python: 3.10.12
- GPU: NVIDIA A40 46 GB
- CUDA used by PyTorch: 12.6
- PyTorch: 2.10.0+cu126
- Transformers: 5.12.1
- PEFT: 0.18.1
- bitsandbytes: 0.49.2
- accelerate: 1.13.0
- flash-attn: 2.8.3+cu126torch2.10
- scikit-learn: 1.7.2
- NumPy: 1.26.4

Base model: `lingshu-medical-mllm/Lingshu-7B`, Hugging Face revision
`b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9`.

Semantic judge: `Qwen/Qwen3-VL-8B-Instruct`, revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.

Exact hashes for the base-model files and VQA-RAD images are under
`provenance/`.
