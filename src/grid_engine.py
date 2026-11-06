"""
Unit 4 Deliverable: Grid Generator Engine (PyTorch / Diffusers)
Sweeps 4 Steps x 3 Guidance Scales with fixed seed, calculates CLIP scores,
and stitches the annotated composite comparison matrix.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.diffusion_pipeline import DiffusionPipelineManager
from src.clip_scorer import CLIPScorer

class ParameterGridGenerator:
    """
    Automated grid generator for parameter exploration.
    Standard sweep:
      - 4 Steps: [5, 15, 25, 50] (or [1, 2, 4, 8] for SD-Turbo)
      - 3 Guidance Scales: [1.5, 7.5, 15.0]
    Annotates each cell with CLIP semantic alignment scores and generation metrics.
    """
    def __init__(
        self,
        pipeline_mgr: Optional[DiffusionPipelineManager] = None,
        clip_scorer: Optional[CLIPScorer] = None
    ):
        self.pipeline_mgr = pipeline_mgr or DiffusionPipelineManager()
        self.clip_scorer = clip_scorer or CLIPScorer()

    def generate_grid(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, deformed, artifacts",
        steps_list: Optional[List[int]] = None,
        guidance_list: Optional[List[float]] = None,
        seed: int = 42,
        cell_size: int = 256,
        progress_callback: Optional[Any] = None
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Runs the 4 steps x 3 guidance sweep, calculates CLIP scores,
        and constructs the composite annotated image.
        """
        if steps_list is None:
            # Default 4 step values representing coarse -> refined convergence
            steps_list = [5, 15, 25, 50]
        if guidance_list is None:
            # Default 3 guidance scale values representing under-conditioned, optimal, and over-saturated
            guidance_list = [1.5, 7.5, 15.0]

        total_cells = len(steps_list) * len(guidance_list)
        results_matrix: List[List[Dict[str, Any]]] = []
        cell_images: List[List[Image.Image]] = []
        
        start_all = time.time()
        completed = 0

        print(f"[GridGenerator] Starting {len(steps_list)}x{len(guidance_list)} sweep for seed={seed}...")

        for r, steps in enumerate(steps_list):
            row_data = []
            row_images = []
            for c, guidance in enumerate(guidance_list):
                img, meta = self.pipeline_mgr.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed,
                    width=cell_size,
                    height=cell_size
                )
                
                # Compute CLIP alignment score
                clip_scores = self.clip_scorer.score(img, prompt)
                clip_score = clip_scores[0] if clip_scores else 25.0
                
                cell_info = {
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed,
                    "clip_score": clip_score,
                    "latency": meta.get("latency_sec", 0.0),
                    "nsfw_filtered": meta.get("nsfw_filtered", False)
                }
                row_data.append(cell_info)
                row_images.append(img)
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_cells, f"Completed Step {steps}, CFG {guidance} (CLIP: {clip_score})")
                    
            results_matrix.append(row_data)
            cell_images.append(row_images)

        total_time = round(time.time() - start_all, 2)
        
        # Stitch composite grid image with labels
        composite_image = self._render_composite_canvas(
            cell_images=cell_images,
            results_matrix=results_matrix,
            steps_list=steps_list,
            guidance_list=guidance_list,
            prompt=prompt,
            seed=seed,
            cell_size=cell_size
        )

        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps_list": steps_list,
            "guidance_list": guidance_list,
            "matrix": results_matrix,
            "total_generation_time_sec": total_time,
            "model_id": self.pipeline_mgr.model_id
        }

        return composite_image, metadata

    def _render_composite_canvas(
        self,
        cell_images: List[List[Image.Image]],
        results_matrix: List[List[Dict[str, Any]]],
        steps_list: List[int],
        guidance_list: List[float],
        prompt: str,
        seed: int,
        cell_size: int
    ) -> Image.Image:
        """
        Renders a publication-ready composite matrix image with column headers,
        row headers, cell-specific CLIP annotations, and title banner.
        """
        num_rows = len(steps_list)
        num_cols = len(guidance_list)
        
        header_height = 110
        row_label_width = 120
        cell_footer_height = 36
        padding = 12
        
        total_width = row_label_width + num_cols * (cell_size + padding) + padding
        total_height = header_height + num_rows * (cell_size + cell_footer_height + padding) + padding
        
        canvas = Image.new("RGB", (total_width, total_height), color=(18, 20, 26))
        draw = ImageDraw.Draw(canvas)
        
        # Title Banner
        title = "P33: Diffusion Parameter Exploration Matrix"
        draw.text((row_label_width, 16), title, fill=(240, 240, 245))
        
        prompt_snippet = prompt if len(prompt) < 85 else prompt[:82] + "..."
        subtitle = f'Prompt: "{prompt_snippet}" | Seed: {seed}'
        draw.text((row_label_width, 42), subtitle, fill=(160, 165, 180))
        
        # Draw Column Headers (Guidance Scales)
        col_labels = [
            f"CFG: {g}\n({'Low / Faded' if g < 3 else ('Optimal' if g < 10 else 'High / Burned')})"
            for g in guidance_list
        ]
        for c, g in enumerate(guidance_list):
            x = row_label_width + c * (cell_size + padding)
            draw.rectangle([x, 70, x + cell_size, 102], fill=(28, 32, 42), outline=(55, 65, 81))
            draw.text((x + 12, 75), f"Guidance: {g} ({'Low' if g < 3 else ('Optimal' if g < 10 else 'High/Oversat')})", fill=(220, 230, 250))

        # Draw Rows and Cells
        for r, steps in enumerate(steps_list):
            y_start = header_height + r * (cell_size + cell_footer_height + padding)
            
            # Row header on left
            draw.rectangle([12, y_start, row_label_width - 8, y_start + cell_size], fill=(28, 32, 42), outline=(55, 65, 81))
            draw.text((20, y_start + cell_size // 2 - 18), f"Steps:\n{steps}", fill=(240, 240, 250))
            
            for c, guidance in enumerate(guidance_list):
                x_start = row_label_width + c * (cell_size + padding)
                img = cell_images[r][c]
                
                # Paste generated cell image
                canvas.paste(img, (x_start, y_start))
                
                # Draw cell outline
                draw.rectangle([x_start, y_start, x_start + cell_size, y_start + cell_size], outline=(60, 70, 90), width=1)
                
                # Cell footer with CLIP Score and Latency
                cell_info = results_matrix[r][c]
                clip_val = cell_info["clip_score"]
                lat_val = cell_info["latency"]
                
                footer_y = y_start + cell_size
                draw.rectangle([x_start, footer_y, x_start + cell_size, footer_y + cell_footer_height], fill=(24, 28, 38))
                
                # Color code CLIP score badge
                clip_color = (74, 222, 128) if clip_val >= 28.0 else ((250, 204, 21) if clip_val >= 24.0 else (248, 113, 113))
                footer_text = f"CLIP: {clip_val:.1f}  |  {lat_val:.2f}s"
                draw.text((x_start + 10, footer_y + 8), footer_text, fill=clip_color)
                
        return canvas

