"""Load the MedGemma baseline with this checkout's Gemma 3 implementation."""

import argparse
import inspect
import json
from pathlib import Path

import torch

import transformers
from transformers import AutoConfig, AutoProcessor, Gemma3ForConditionalGeneration


def load_medgemma(model_id, revision="main", device="cpu", dtype="float32", local_files_only=False):
    """Load a matching config, processor and model; reject incomplete weight loading."""
    expected_source = Path(__file__).resolve().parents[2] / "src" / "transformers"
    if Path(transformers.__file__).resolve().parent != expected_source:
        raise RuntimeError("Run from the repository root with PYTHONPATH=src to use this checkout.")

    source = {"revision": revision, "local_files_only": local_files_only}
    config = AutoConfig.from_pretrained(model_id, **source)
    if config.model_type != "gemma3":
        raise ValueError(f"Expected a gemma3 checkpoint, got {config.model_type!r}")
    # Pin the processor and weights to the same resolved Hub commit as the config.
    source["revision"] = getattr(config, "_commit_hash", None) or revision
    processor = AutoProcessor.from_pretrained(model_id, **source)
    model, loading_info = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        config=config,
        dtype=getattr(torch, dtype),
        device_map=device,
        use_safetensors=True,
        output_loading_info=True,
        **source,
    )
    problems = {key: value for key, value in loading_info.items() if value}
    if problems:
        raise RuntimeError(f"Baseline weights did not load cleanly: {problems}")
    model.eval()
    print(
        json.dumps(
            {
                "checkpoint": str(model_id),
                "revision": source["revision"],
                "transformers_version": transformers.__version__,
                "implementation": inspect.getfile(type(model)),
                "model_class": type(model).__name__,
                "dtype": str(model.dtype),
                "device_map": {key: str(value) for key, value in model.hf_device_map.items()},
                "loading_info": {key: list(value) for key, value in loading_info.items()},
            },
            indent=2,
        )
    )
    return model, processor


@torch.inference_mode()
def generate(model, processor, prompt, image_path=None, max_new_tokens=128):
    """Run a single text or image/text prompt with the checkpoint's chat template."""
    content = []
    if image_path is not None:
        from PIL import Image

        with Image.open(image_path) as image:
            content.append({"type": "image", "image": image.convert("RGB")})
    content.append({"type": "text", "text": prompt})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = outputs[0, inputs["input_ids"].shape[-1] :]
    answer = processor.decode(generated_ids, skip_special_tokens=True)
    print(f"[generation] Finished: {generated_ids.numel()} new tokens.", flush=True)
    if not answer.strip():
        raw_text = processor.decode(generated_ids, skip_special_tokens=False)
        raise RuntimeError(
            f"Generation returned no visible text. Token IDs: {generated_ids.tolist()}; raw text: {raw_text!r}"
        )
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--revision", default="main", help="Use a Hub commit SHA for reproducible experiments.")
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda:0, or auto (Accelerate inference dispatch).")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", help="If omitted, only load and validate the checkpoint.")
    parser.add_argument("--image", type=Path, help="Optional local image for the prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.image is not None and (not args.image.is_file() or not args.prompt):
        parser.error("--image requires an existing image file and --prompt")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must contain non-whitespace text")
    print(f"[run] mode={'generate' if args.prompt is not None else 'load-only'}", flush=True)
    model, processor = load_medgemma(args.model_id, args.revision, args.device, args.dtype, args.local_files_only)
    if args.prompt is not None:
        print("[generation] Starting prompt processing and generation...", flush=True)
        print(generate(model, processor, args.prompt, args.image, args.max_new_tokens))
    else:
        print("[run] Load complete. No --prompt supplied; generation skipped.", flush=True)


if __name__ == "__main__":
    main()
