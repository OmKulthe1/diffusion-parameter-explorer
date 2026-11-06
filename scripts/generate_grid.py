"""
CLI Deliverable: Generate 4 Steps x 3 Guidance Grid
Usage:
  python scripts/generate_grid.py --prompt "cinematic cyberpunk samurai in neon rain" --seed 42
"""

import os
import argparse
import json
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid_engine import ParameterGridGenerator
from src.diffusion_pipeline import DiffusionPipelineManager
from src.clip_scorer import CLIPScorer

def main():
    parser = argparse.ArgumentParser(description="Generate 4 Steps x 3 Guidance Diffusion Grid")
    parser.add_argument(
        "--prompt",
        type=str,
        default="cinematic medium shot of a cyberpunk samurai standing in rain-slicked Tokyo street, glowing neon blue katana, reflective puddle ripples, 8k resolution, photorealistic",
        help="Prompt for diffusion grid generation"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, deformed, low quality, bad anatomy, text, watermark",
        help="Negative prompt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed to isolate parameter impact"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/sd-turbo",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Use deterministic fast offline diffusion engine"
    )
    parser.add_argument(
        "--online",
        dest="offline",
        action="store_false",
        help="Download online weights from Hugging Face"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/grid_4x3_showcase.png",
        help="Output image path"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("=" * 70)
    print("  P33: Diffusion Parameter Explorer - 4x3 Grid Generator")
    print("=" * 70)
    print(f"Model: {args.model} ({'Offline fast mode' if args.offline else 'Online weights'})")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}")
    print(f"Steps: [5, 15, 25, 50]")
    print(f"Guidance Scales: [1.5, 7.5, 15.0]")
    print("-" * 70)

    pipe_mgr = DiffusionPipelineManager(model_id=args.model, force_offline_engine=args.offline)
    clip_scorer = CLIPScorer()
    grid_gen = ParameterGridGenerator(pipeline_mgr=pipe_mgr, clip_scorer=clip_scorer)

    def on_step(completed, total, msg):
        print(f"  [{completed}/{total}] {msg}")

    grid_img, meta = grid_gen.generate_grid(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps_list=[5, 15, 25, 50],
        guidance_list=[1.5, 7.5, 15.0],
        seed=args.seed,
        cell_size=256,
        progress_callback=on_step
    )

    grid_img.save(args.output)
    print("-" * 70)
    print(f"[SUCCESS] Grid image saved to: {args.output}")

    meta_path = os.path.splitext(args.output)[0] + "_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[SUCCESS] Grid metadata saved to: {meta_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

