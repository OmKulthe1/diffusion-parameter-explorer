"""
Unit 1 CLI Deliverable: Noise Distribution Analysis & Prompt Variant Generator
Demonstrates Gaussian noise properties, forward diffusion schedule trajectories,
and prints structured prompt engineering variants.
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.noise_theory import (
    sample_gaussian_noise,
    compute_distribution_metrics,
    get_beta_schedule,
    simulate_diffusion_trajectory
)
from src.prompt_variants import CURATED_PROMPTS, generate_prompt_variant

def main():
    os.makedirs("output", exist_ok=True)
    print("=" * 75)
    print("  Unit 1: Noise as a Probability Distribution & Prompt Variants")
    print("=" * 75)

    # 1. Mathematical Analysis of Standard Gaussian Noise
    print("\n[Part 1] Analyzing Standard Gaussian Noise N(0, I) in Latent Space (4 x 64 x 64 = 16,384 dims):")
    latent_shape = (4, 64, 64)
    noise_sample = sample_gaussian_noise(latent_shape, seed=42)
    metrics = compute_distribution_metrics(noise_sample)

    print(f"  Sample Dimensions:       {latent_shape} = {np.prod(latent_shape)} latent dimensions")
    print(f"  Empirical Mean:          {metrics['mean']:.5f} (Theoretical: 0.00000)")
    print(f"  Empirical Variance:      {metrics['variance']:.5f} (Theoretical: 1.00000)")
    print(f"  Empirical Std Dev:       {metrics['std']:.5f} (Theoretical: 1.00000)")
    print(f"  Skewness:                {metrics['skewness']:.5f} (Theoretical: 0.00000 -> Perfect symmetry)")
    print(f"  Excess Kurtosis:         {metrics['excess_kurtosis']:.5f} (Theoretical: 0.00000 -> Mesokurtic Gaussian)")
    print(f"  Sample Extrema:          [{metrics['min']:.3f}, {metrics['max']:.3f}]")

    # 2. Forward Diffusion Trajectory Visualization
    print("\n[Part 2] Simulating Forward Diffusion Markov Chain:")
    print("  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)")
    
    # Create synthetic reference image (256x256 test pattern)
    size = 256
    y, x = np.mgrid[0:size, 0:size]
    test_img = np.zeros((size, size, 3), dtype=np.float32)
    # Gradient circle pattern
    r = np.sqrt((x - size/2)**2 + (y - size/2)**2)
    test_img[:, :, 0] = np.clip(1.0 - r / (size * 0.4), 0, 1)
    test_img[:, :, 1] = np.clip(np.sin(x / 20.0) * 0.5 + 0.5, 0, 1)
    test_img[:, :, 2] = np.clip(np.cos(y / 20.0) * 0.5 + 0.5, 0, 1)
    
    # Normalize to [-1, 1] for diffusion equations
    x_0 = (test_img * 2.0 - 1.0).astype(np.float32)
    
    timesteps = [0, 100, 250, 500, 750, 999]
    trajectory = simulate_diffusion_trajectory(x_0, timesteps=timesteps, schedule_type="scaled_linear", seed=42)
    
    # Render composite strip of forward noising
    strip_w = len(timesteps) * size + (len(timesteps) + 1) * 12
    strip_h = size + 80
    strip = Image.new("RGB", (strip_w, strip_h), (18, 20, 26))
    draw = ImageDraw.Draw(strip)
    draw.text((16, 12), "Unit 1: Forward Diffusion Noising Process q(x_t | x_0) across Timesteps t", fill=(240, 245, 255))

    for idx, (t_step, x_t) in enumerate(trajectory):
        # Convert [-1, 1] back to [0, 255]
        img_np = np.clip((x_t + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        pos_x = 12 + idx * (size + 12)
        pos_y = 45
        strip.paste(pil_img, (pos_x, pos_y))
        draw.rectangle([pos_x, pos_y, pos_x + size, pos_y + size], outline=(60, 70, 90), width=1)
        
        # Label below
        draw.text((pos_x + 10, pos_y + size - 24), f"t = {t_step}", fill=(255, 255, 255))
        
    out_trajectory_path = "output/noise_trajectory.png"
    strip.save(out_trajectory_path)
    print(f"  Forward diffusion strip saved to: {out_trajectory_path}")

    # 3. Prompt Engineering Variants Demonstration
    print("\n[Part 3] Systematic Prompt Variants Library:")
    for category, variants in CURATED_PROMPTS.items():
        print(f"\n  --- Category: {category} ---")
        for v in variants:
            print(f"    * Name: {v.name}")
            print(f"      Positive: {v.positive_prompt[:75]}...")
            print(f"      Negative: {v.negative_prompt[:60]}...")
            print(f"      Recommended: Steps={v.recommended_steps}, CFG={v.recommended_guidance}")
            print(f"      Rationale: {v.rationale}")

    # Generate custom variant demo
    custom_variant = generate_prompt_variant(
        subject="a robotic hummingbird hovering near a mechanical orchid",
        medium_style="cinematic",
        lighting="golden hour rim lighting",
        camera_view="macro close-up",
        mood="mystical"
    )
    print("\n[Custom Generated Variant Sample]:")
    print(f"  Positive: {custom_variant['positive']}")
    print(f"  Negative: {custom_variant['negative']}")
    print("=" * 75)

if __name__ == "__main__":
    main()

