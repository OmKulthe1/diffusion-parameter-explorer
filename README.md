# P33 · Diffusion Parameter Explorer

An end-to-end interactive exploration and benchmarking suite demonstrating how **inference steps ($T$)**, **classifier-free guidance scale ($s$)**, **random seed**, and **negative prompts** govern the sampling trajectories, semantic alignment, and visual characteristics of diffusion models.

Developed as a complete implementation of **Project P33**, covering all 5 core units and all required deliverables.

---

## 🎯 Project Goals & Units Covered

1. **U1 — Explain Noise as a Probability Distribution; Write Prompt Variants**
   - Mathematical analysis of isotropic multivariate Gaussian noise $\mathcal{N}(0, I)$ in latent space ($4 \times 64 \times 64 = 16,384$ dimensions).
   - High-dimensional concentration of measure ("Gaussian soap bubble" theorem).
   - Variance schedules: Linear (DDPM), Scaled Linear (Latent Diffusion), and Cosine (Nichol & Dhariwal).
   - Closed-form forward diffusion Markov chain: $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$.
   - Systematic prompt engineering variant engine with modular subject, style, lighting, camera, and quality modifiers.

2. **U2 — Generate DCGAN Samples to Compare with Diffusion**
   - Pure PyTorch DCGAN Generator & Discriminator implementation following Radford et al. (2015).
   - Transposed convolution deconvolution dynamics and checkerboard artifact analysis.
   - 7-dimension architectural, operational, and mathematical comparison against diffusion models.

3. **U3 — Run Stable Diffusion / SD-Turbo; Score Images with CLIP**
   - Hugging Face `diffusers` integration supporting `stabilityai/sd-turbo` and `runwayml/stable-diffusion-v1-5`.
   - VRAM optimizations tailored for NVIDIA GeForce RTX 3050 (4GB) via FP16 precision, attention slicing, and VAE tiling.
   - Real-time CLIP ViT semantic alignment scoring: $\text{CLIP Score} = 100 \times \max(\cos(E_I(x), E_T(y)), 0)$.

4. **U4 — Build the Grid Generator with Diffusers (PyTorch)**
   - Automated grid generator executing a $4 \times 3$ matrix sweep:
     - **4 Steps**: $5, 15, 25, 50$ (or $1, 2, 4, 8$ for SD-Turbo)
     - **3 Guidance Scales**: $1.5$ (Low/Faded), $7.5$ (Optimal), $15.0$ (High/Oversaturated)
     - **Fixed Seed**: Isolates parameter impact from initial noise variance.
   - Stitches high-resolution composite canvas with row/col headers, per-cell CLIP score badges, and latencies.

5. **U5 — Enable NSFW Filter; Check Model License & Representation**
   - Official cosine concept suppression safety filter.
   - CreativeML Open RAIL-M license audit: permitted rights, downstream inheritance, and Attachment A prohibited uses.
   - Comprehensive demographic, occupational, and geographic representation audit on LAION-5B with mitigation strategies.

---

## 📦 Project Structure

```
diffusion-parameter-explorer/
├── .venv/                          # Dedicated Python virtual environment
├── app.py                          # Flagship Streamlit interactive web application
├── requirements.txt                # Python package dependencies
├── README.md                       # Master documentation & quickstart
├── src/
│   ├── __init__.py
│   ├── noise_theory.py             # U1: Gaussian math, beta schedules, SDE forward noising
│   ├── prompt_variants.py          # U1: Modular prompt builder & curated library
│   ├── dcgan.py                    # U2: Radford DCGAN architecture & sampling
│   ├── diffusion_pipeline.py       # U3 & U5: SD / SD-Turbo manager, memory optimization, safety filter
│   ├── clip_scorer.py              # U3: Hugging Face CLIP ViT cosine alignment scorer
│   ├── grid_engine.py              # U4: 4x3 parameter matrix generator & composite stitcher
│   └── safety_and_ethics.py        # U5: NSFW filter, Open RAIL-M audit, demographic bias study
├── scripts/
│   ├── run_app.bat                 # 1-click Windows launcher for web app
│   ├── generate_grid.py            # CLI deliverable: 4x3 parameter grid generator
│   ├── compare_gan_diffusion.py    # CLI deliverable: DCGAN vs Diffusion benchmark
│   └── explain_noise.py            # CLI deliverable: Noise distribution & prompt variants
├── output/
│   ├── grid_4x3_showcase.png       # Deliverable: Annotated 4x3 parameter matrix
│   ├── grid_4x3_showcase_metadata.json
│   ├── gan_vs_diffusion.png        # Deliverable: DCGAN vs Diffusion benchmark canvas
│   ├── gan_vs_diffusion_metrics.json
│   └── noise_trajectory.png        # U1: Forward noising Markov chain strip
└── docs/
    └── REPORT.md                   # Comprehensive technical report
```

---

## 🚀 Quickstart Guide

### 1. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Run the Interactive Web Explorer App (Deliverable 1)
```powershell
streamlit run app.py
```
Or simply double-click `scripts\run_app.bat`.

### 3. Generate the 4 Steps × 3 Guidance Grid (Deliverable 2)
```powershell
python scripts/generate_grid.py --prompt "cinematic medium shot of a cyberpunk samurai standing in rain-slicked Tokyo street, glowing neon blue katana, reflective puddle ripples, 8k resolution" --seed 42
```
Output saved to `output/grid_4x3_showcase.png`.

### 4. Run the GAN vs Diffusion Benchmark (Deliverable 3)
```powershell
python scripts/compare_gan_diffusion.py
```
Output saved to `output/gan_vs_diffusion.png`.

### 5. Inspect Noise Distributions & Schedules (Unit 1)
```powershell
python scripts/explain_noise.py
```
Output saved to `output/noise_trajectory.png`.

---

## 📊 Key Experimental Findings

1. **Inference Steps ($T$)**:
   - $T < 10$: Initial low-frequency semantic structures are formed, but fine details (hair, reflections, textural grain) remain blurred and unresolved.
   - $T \approx 25$: Sweet spot for standard Stable Diffusion; yields sharp edges, photorealistic textures, and high CLIP alignment scores.
   - $T > 40$: Diminishing returns; perceptual quality saturates while inference time scales linearly.
   - For **SD-Turbo**, ADD distillation achieves near-optimal convergence in only $1 - 4$ steps.

2. **Classifier-Free Guidance ($s$)**:
   - $s < 3.0$: Images appear washed-out, pastel, and low-contrast, failing to capture specific prompt tokens.
   - $s \in [7.0, 9.0]$: Optimal regime; strong semantic faithfulness, vibrant dynamic range, and sharp boundary contrast.
   - $s > 14.0$: Over-saturation regime; pushes latents beyond the training manifold, resulting in severe color burn, haloing, and high-frequency chromatic aberrations.

3. **Negative Prompting**:
   - Acts as a repulsive conditioning vector in CFG:
     $$\hat{\epsilon} = \epsilon_\theta(z_t, \varnothing) + s \cdot (\epsilon_\theta(z_t, c_+) - \epsilon_\theta(z_t, c_-))$$
   - Effectively removes unwanted color casts, extra limbs, and textural blur without changing the underlying composition dictated by the random seed.
