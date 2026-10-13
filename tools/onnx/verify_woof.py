#!/usr/bin/env python3
"""Compare the original MLX checkpoint and reconstructed HF text weights.

This is an export diagnostic, not part of Rust inference. Uses deterministic
unconstrained JSON requests, and compares last-position logits before decoding.
"""
import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", help="also compare an exported GenAI bundle")
    args = parser.parse_args()
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    original, mlx_tokenizer = load(args.mlx)
    hf = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    system = "You fill the selected browser field from the user's goal and page context. Return only a JSON object with a text property containing the value to type. Use only the requested component: a first-name field needs the first name, not the full name. Never invent personal details."
    for label, goal in [("First name", "Fill this form for Alice Chen."),
                        ("Email address", "Sign up using alice@example.com.")]:
        user = json.dumps({"goal": goal, "selected_field": {"label": label, "tag": "input", "current_value": ""},
                           "page_context": {"title": "Contact details", "fields": ["First name", "Email address", "City"]}},
                          sort_keys=True, separators=(",", ":"))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        mlx_ids = mlx_tokenizer.encode(prompt, add_special_tokens=False)
        assert ids == mlx_ids, "Tokenizers differ"
        reference = np.array(original(mx.array([ids]))[0, -1].astype(mx.float32))
        with torch.no_grad():
            candidate = hf(torch.tensor([ids])).logits[0, -1].numpy()
        print(label, "logits MAE", float(np.abs(reference - candidate).mean()),
              "MLX top5", np.argsort(reference)[-5:][::-1], "HF top5", np.argsort(candidate)[-5:][::-1], flush=True)
        print("MLX:", generate(original, mlx_tokenizer, prompt=prompt, max_tokens=64,
                               sampler=make_sampler(temp=0)), flush=True)
        with torch.no_grad():
            output = hf.generate(torch.tensor([ids]), max_new_tokens=64, do_sample=False)
        print("HF:", tokenizer.decode(output[0, len(ids):], skip_special_tokens=True), flush=True)
        if args.onnx:
            import onnxruntime_genai as og
            native = og.Model(args.onnx)
            native_tokenizer = og.Tokenizer(native)
            for constrained in (False, True):
                params = og.GeneratorParams(native)
                params.set_search_options(do_sample=False, max_length=len(ids) + 64)
                if constrained:
                    params.set_guidance("json_schema", json.dumps({"type":"object", "properties":{"text":{"type":"string"}}, "required":["text"]}), False)
                generator = og.Generator(native, params)
                generator.append_tokens(ids)
                stream = native_tokenizer.create_stream()
                text = ""
                while not generator.is_done():
                    generator.generate_next_token()
                    text += stream.decode(int(generator.get_next_tokens()[0]))
                print("ONNX constrained=" + str(constrained) + ": " + text, flush=True)
                del generator, params


if __name__ == "__main__":
    main()
