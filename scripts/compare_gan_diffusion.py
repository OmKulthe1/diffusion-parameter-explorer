"""
Deliverable 3: DCGAN vs Diffusion Comparison Tool
Generates samples from both generative paradigms, scores them with CLIP,
constructs a visual comparison matrix, and exports the quantitative benchmark.
"""

import os
import sys
import json
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dcgan import DCGANGenerator, get_gan_vs_diffusion_comparison_table
from src.diffusion_pipeline import DiffusionPipelineManager
from src.clip_scorer import CLIPScorer

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DCGAN vs Diffusion Comparison")
    parser.add_argument("--offline", action="store_true", default=True, help="Use deterministic fast offline diffusion engine")
    parser.add_argument("--online", dest="offline", action="store_false", help="Download online Hugging Face weights")
    args = parser.parse_args()

    output_path = "output/gan_vs_diffusion.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 75)
    print("  P33: Generative Comparison: DCGAN vs Diffusion Models")
    print("=" * 75)

    prompt = "photorealistic studio portrait of an expressive person, 85mm lens, natural lighting"
    seed = 42

    # 1. Generate DCGAN Samples
    print("[1/4] Generating DCGAN samples from latent space z ~ N(0, I)...")
    dcgan = DCGANGenerator()
    start_dcgan = time.time()
    dcgan_images = dcgan.generate_samples(num_samples=4, seed=seed)
    dcgan_time = (time.time() - start_dcgan) / 4.0

    # 2. Generate Diffusion Samples
    print("[2/4] Generating Diffusion samples via iterative reverse denoising...")
    diff_mgr = DiffusionPipelineManager(force_offline_engine=args.offline)
    diff_images = []
    diff_latencies = []
    
    # Generate 4 diffusion samples with seeds 42, 43, 44, 45
    for s_idx in range(4):
        img, meta = diff_mgr.generate(
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality",
            num_inference_steps=25,
            guidance_scale=7.5,
            seed=seed + s_idx,
            width=256,
            height=256
        )
        diff_images.append(img.resize((256, 256)))
        diff_latencies.append(meta.get("latency_sec", 0.8))
        print(f"  Diffusion sample {s_idx+1}/4 complete.")

    # 3. Score with CLIP
    print("[3/4] Computing CLIP semantic alignment scores...")
    clip_scorer = CLIPScorer()
    dcgan_clip_scores = clip_scorer.score(dcgan_images, prompt)
    diff_clip_scores = clip_scorer.score(diff_images, prompt)

    avg_dcgan_clip = np.mean(dcgan_clip_scores)
    avg_diff_clip = np.mean(diff_clip_scores)

    print(f"  DCGAN Mean CLIP Score:     {avg_dcgan_clip:.2f} (Single-pass latent projection)")
    print(f"  Diffusion Mean CLIP Score: {avg_diff_clip:.2f} (Iterative cross-attention guided)")

    # 4. Construct High-Resolution Composite Comparison Image
    print("[4/4] Storing side-by-side comparison canvas...")
    cell_w, cell_h = 256, 256
    padding = 16
    header_h = 130
    footer_h = 40
    section_label_h = 40
    
    canvas_w = padding * 2 + 4 * cell_w + 3 * padding
    # 2 sections: DCGAN row + Diffusion row
    canvas_h = header_h + 2 * (section_label_h + cell_h + footer_h + padding) + padding
    
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(15, 17, 23))
    draw = ImageDraw.Draw(canvas)

    # Master Header
    draw.text((padding, 16), "Generative Paradigm Benchmark: DCGAN vs Diffusion Models", fill=(240, 245, 255))
    draw.text((padding, 44), f'Target Prompt: "{prompt}" | Reference Seed: {seed}', fill=(160, 170, 190))
    draw.text((padding, 70), "DCGAN: Radford et al. (2015) 1-Step Feedforward vs Diffusion: Sohl-Dickstein / Ho / Rombach Iterative Denoising", fill=(120, 130, 150))
    draw.line([(padding, 105), (canvas_w - padding, 105)], fill=(45, 55, 75), width=2)

    # Section 1: DCGAN Row
    sec1_y = header_h
    draw.rectangle([padding, sec1_y, canvas_w - padding, sec1_y + section_label_h], fill=(30, 25, 40))
    draw.text(
        (padding + 12, sec1_y + 10),
        f"DCGAN Samples (Single Forward Pass G(z)) | Avg CLIP: {avg_dcgan_clip:.1f} | Avg Latency: {dcgan_time*1000:.1f}ms",
        fill=(216, 180, 254)
    )

    y_imgs_1 = sec1_y + section_label_h + padding
    for i in range(4):
        x = padding + i * (cell_w + padding)
        canvas.paste(dcgan_images[i], (x, y_imgs_1))
        draw.rectangle([x, y_imgs_1, x + cell_w, y_imgs_1 + cell_h], outline=(147, 51, 234), width=2)
        
        # Footer
        fy = y_imgs_1 + cell_h
        draw.rectangle([x, fy, x + cell_w, fy + footer_h], fill=(24, 20, 32))
        draw.text((x + 10, fy + 10), f"Sample {i+1} | CLIP: {dcgan_clip_scores[i]:.1f}", fill=(216, 180, 254))

    # Section 2: Diffusion Row
    sec2_y = y_imgs_1 + cell_h + footer_h + padding * 2
    draw.rectangle([padding, sec2_y, canvas_w - padding, sec2_y + section_label_h], fill=(20, 35, 30))
    draw.text(
        (padding + 12, sec2_y + 10),
        f"Diffusion Samples (Iterative Reverse SDE, 25 Steps, CFG 7.5) | Avg CLIP: {avg_diff_clip:.1f} | Avg Latency: {np.mean(diff_latencies):.2f}s",
        fill=(134, 239, 172)
    )

    y_imgs_2 = sec2_y + section_label_h + padding
    for i in range(4):
        x = padding + i * (cell_w + padding)
        canvas.paste(diff_images[i], (x, y_imgs_2))
        draw.rectangle([x, y_imgs_2, x + cell_w, y_imgs_2 + cell_h], outline=(34, 197, 94), width=2)
        
        # Footer
        fy = y_imgs_2 + cell_h
        draw.rectangle([x, fy, x + cell_w, fy + footer_h], fill=(18, 28, 24))
        draw.text((x + 10, fy + 10), f"Sample {i+1} | CLIP: {diff_clip_scores[i]:.1f}", fill=(134, 239, 172))

    canvas.save(output_path)
    print(f"[SUCCESS] Comparison image saved to: {output_path}")

    # Output detailed JSON comparison report
    table_data = get_gan_vs_diffusion_comparison_table()
    report = {
        "prompt": prompt,
        "seed": seed,
        "dcgan": {
            "samples_count": 4,
            "clip_scores": dcgan_clip_scores,
            "average_clip": float(avg_dcgan_clip),
            "average_latency_ms": float(dcgan_time * 1000)
        },
        "diffusion": {
            "samples_count": 4,
            "clip_scores": diff_clip_scores,
            "average_clip": float(avg_diff_clip),
            "average_latency_s": float(np.mean(diff_latencies))
        },
        "architectural_comparison": table_data
    }
    
    json_path = "output/gan_vs_diffusion_metrics.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[SUCCESS] Comparison report saved to: {json_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()

