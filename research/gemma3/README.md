# MedGemma 1.5 baseline using local Gemma 3

Run all commands from the repository root. The loader uses the actual
`Gemma3ForConditionalGeneration` implementation in this checkout and rejects
an import from an unrelated Transformers installation. It does not modify the architecture.

Install dependencies in your research environment:

```bash
python -m pip install -e .
python -m pip install torch accelerate pillow
```

Accept access terms on https://huggingface.co/google/medgemma-1.5-4b-it and
run `hf auth login`, or supply `HF_TOKEN` through your environment. Do not put
tokens in source code. The first run downloads the checkpoint into the normal
Hugging Face cache.

Load and validate on CPU (float32; roughly 16 GB for weights alone, plus runtime memory):

```bash
PYTHONPATH=src python research/gemma3/load_medgemma.py
```

Apple Silicon, using float16 to reduce memory:

```bash
PYTHONPATH=src python research/gemma3/load_medgemma.py \
  --device mps --dtype float16 --prompt "What is the function of red blood cells?"
```

CUDA GPU with bfloat16 support:

```bash
PYTHONPATH=src python research/gemma3/load_medgemma.py \
  --device cuda:0 --dtype bfloat16 \
  --image /path/to/image.png --prompt "Describe this image." --max-new-tokens 128
```

Use `--device auto` for Accelerate inference placement across available devices.
This is an inference loader, not a distributed training setup. CPU float32 is
the conservative default; accelerator dtype support and available memory depend
on your machine. Text-only generation does not exercise the vision encoder.

The loader obtains config, processor and weights from the same checkpoint revision,
uses safetensors, and fails on missing/unexpected weights or incompatible shapes.
It prints the implementation path, resolved revision and loading diagnostics.
For reproducibility, pass `--revision COMMIT_SHA` and record the local Git commit,
environment versions and output alongside each experiment. Use `--local-files-only`
for a cached checkpoint, or `--model-id /path/to/checkpoint` for local files.

Keep this strict loader as your baseline. When experimenting, edit
`src/transformers/models/gemma3/modular_gemma3.py` and run `make fix-repo` to regenerate
the runtime implementation, or create a separate research model class. If a new
architecture changes parameter names/shapes, explicitly map and initialize those
parameters in a separate loader rather than suppressing baseline loading errors.

Model documentation: https://huggingface.co/google/medgemma-1.5-4b-it
