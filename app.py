"""
P33 · Diffusion Parameter Explorer
Deliverable 1: Comprehensive Interactive Streamlit Web Application
Covers all Units (U1 to U5) and Deliverables:
  - U1: Noise as Probability Distribution & Prompt Variants
  - U2: DCGAN Samples & Structural Comparison with Diffusion
  - U3: Stable Diffusion / SD-Turbo & Real-time CLIP Scoring
  - U4: 4 Steps x 3 Guidance Grid Generator
  - U5: NSFW Safety Filter & CreativeML Open RAIL-M License & Representation Audit
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import time
import json
import matplotlib.pyplot as plt

# Internal modular imports
from src.noise_theory import (
    sample_gaussian_noise,
    compute_distribution_metrics,
    get_beta_schedule,
    simulate_diffusion_trajectory
)
from src.prompt_variants import CURATED_PROMPTS, generate_prompt_variant
from src.dcgan import DCGANGenerator, get_gan_vs_diffusion_comparison_table
from src.diffusion_pipeline import DiffusionPipelineManager
from src.clip_scorer import CLIPScorer
from src.grid_engine import ParameterGridGenerator
from src.safety_and_ethics import (
    SafetyChecker,
    get_open_rail_m_license_summary,
    get_representation_and_bias_audit
)

# Page configuration
st.set_page_config(
    page_title="Diffusion Parameter Explorer | P33",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global custom CSS for modern GenAI dashboard styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #334155;
    }
    .badge-optimal {
        background-color: #065f46;
        color: #6ee7b7;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .badge-warn {
        background-color: #78350f;
        color: #fde68a;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App Navigation
st.markdown('<div class="main-title">P33 · Diffusion Parameter Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive exploration of Steps, Guidance Scale, Seed, and Negative Prompts with CLIP scoring, DCGAN benchmarks & Ethics</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🎨 Parameter Playground",
    "📊 4x3 Parameter Grid",
    "⚔️ DCGAN vs Diffusion",
    "📐 Noise & Schedules (U1)",
    "⚖️ Safety & Representation (U5)"
])

# Initialize cached instances in session state
if "pipeline_mgr" not in st.session_state:
    st.session_state.pipeline_mgr = DiffusionPipelineManager(model_id="stabilityai/sd-turbo")
if "clip_scorer" not in st.session_state:
    st.session_state.clip_scorer = CLIPScorer()
if "dcgan_gen" not in st.session_state:
    st.session_state.dcgan_gen = DCGANGenerator()

# ==============================================================================
# TAB 1: PARAMETER PLAYGROUND (U3 & Core Goal)
# ==============================================================================
with tabs[0]:
    st.subheader("Interactive Parameter Explorer (Steps, CFG, Seed, Negative Prompt)")
    
    col_ctrl, col_view = st.columns([1, 1.2], gap="large")
    
    with col_ctrl:
        st.markdown("#### 1. Conditioning & Prompts")
        
        # Curated presets picker
        prompt_categories = list(CURATED_PROMPTS.keys())
        sel_cat = st.selectbox("📂 Prompt Category Preset", prompt_categories)
        cat_variants = CURATED_PROMPTS[sel_cat]
        sel_var = st.selectbox("🎯 Select Preset Variant", cat_variants, format_func=lambda v: v.name)
        
        prompt_input = st.text_area(
            "Positive Prompt",
            value=sel_var.positive_prompt,
            height=90,
            help="Text description conditioning the cross-attention layers of the U-Net"
        )
        
        neg_prompt_input = st.text_area(
            "Negative Prompt (Suppressed Concepts)",
            value=sel_var.negative_prompt,
            height=70,
            help="Steers Classifier-Free Guidance (CFG) away from undesired visual traits"
        )
        
        st.markdown("#### 2. Sampling Hyperparameters")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            engine_mode = st.selectbox(
                "Execution Mode",
                ["High-Fidelity Denoising Engine (Instant)", "Online Hugging Face Pipeline (Download weights)"],
                index=0
            )
            model_opt = st.selectbox(
                "Diffusion Architecture",
                ["stabilityai/sd-turbo (Fast 1-4 steps)", "runwayml/stable-diffusion-v1-5 (20-50 steps)"],
                index=0
            )
        with col_m2:
            safety_toggle = st.toggle("NSFW Safety Filter (U5)", value=True, help="Official safety checker suppresses sensitive concept cosine matches")
            st.caption("🛡️ Safety filter inspects CLIP concept embeddings and blanks out violating outputs.")
            
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            is_turbo = "turbo" in model_opt.lower()
            default_steps = 4 if is_turbo else 25
            max_steps = 10 if is_turbo else 50
            steps_slider = st.slider(
                "Inference Steps (T)",
                min_value=1,
                max_value=max_steps,
                value=default_steps,
                help="Number of reverse denoising iterations. Higher = cleaner details; Diminishing returns after optimal point."
            )
        with col_p2:
            default_cfg = 2.0 if is_turbo else 7.5
            cfg_slider = st.slider(
                "Guidance Scale (CFG)",
                min_value=0.0,
                max_value=20.0,
                value=default_cfg,
                step=0.5,
                help="Classifier-Free Guidance weight (s). <3.0: washed out; 7.0-9.0: balanced; >14.0: oversaturated/color-burn."
            )
            
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            seed_val = st.number_input("Random Seed", min_value=0, max_value=9999999, value=42, step=1)
        with col_s2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🎲 Randomize"):
                seed_val = int(np.random.randint(0, 9999999))
                st.rerun()

        gen_btn = st.button("🚀 Generate Diffusion Image", type="primary", use_container_width=True)

    with col_view:
        st.markdown("#### Output & Real-Time Alignment")
        
        if gen_btn or "last_gen_img" not in st.session_state:
            with st.spinner("Executing reverse diffusion trajectory..."):
                model_id_clean = "stabilityai/sd-turbo" if "turbo" in model_opt.lower() else "runwayml/stable-diffusion-v1-5"
                st.session_state.pipeline_mgr.model_id = model_id_clean
                st.session_state.pipeline_mgr.use_safety_checker = safety_toggle
                st.session_state.pipeline_mgr.force_offline_engine = ("Instant" in engine_mode)
                
                img_out, meta_out = st.session_state.pipeline_mgr.generate(
                    prompt=prompt_input,
                    negative_prompt=neg_prompt_input,
                    num_inference_steps=steps_slider,
                    guidance_scale=cfg_slider,
                    seed=seed_val,
                    width=400,
                    height=400
                )
                
                # Real-time CLIP Scoring
                clip_scores = st.session_state.clip_scorer.score(img_out, prompt_input)
                clip_score = clip_scores[0]
                
                st.session_state.last_gen_img = img_out
                st.session_state.last_gen_meta = meta_out
                st.session_state.last_gen_clip = clip_score

        # Display output image
        st.image(st.session_state.last_gen_img, caption=f"Generated Output (Steps: {steps_slider}, CFG: {cfg_slider}, Seed: {seed_val})", use_container_width=True)
        
        # Display Metrics Row
        m_col1, m_col2, m_col3 = st.columns(3)
        clip_val = st.session_state.last_gen_clip
        with m_col1:
            st.metric("CLIP Semantic Score", f"{clip_val:.2f} / 100", help="Cosine similarity of image & text in CLIP ViT feature space")
        with m_col2:
            st.metric("Latency", f"{st.session_state.last_gen_meta['latency_sec']}s")
        with m_col3:
            nsfw_status = "🛡️ Safe" if not st.session_state.last_gen_meta.get("nsfw_filtered") else "⚠️ Censored"
            st.metric("Safety Status", nsfw_status)
            
        # Guidance Scale Regime Feedback
        if cfg_slider < 3.0:
            st.info("ℹ️ **Low Guidance (< 3.0)**: Image is soft, poetic, but weakly aligned with prompt specifics.")
        elif 6.5 <= cfg_slider <= 9.0:
            st.success("✅ **Optimal Guidance (6.5 - 9.0)**: Strong balance between semantic adherence, sharp contrast, and natural textures.")
        elif cfg_slider > 13.0:
            st.warning("⚠️ **Extreme Guidance (> 13.0)**: High probability of chromatic burn, oversaturation, and boundary distortion artifacts.")

# ==============================================================================
# TAB 2: 4x3 PARAMETER GRID (Deliverable 2 & U4)
# ==============================================================================
with tabs[1]:
    st.subheader("4 Steps × 3 Guidance Grid Generator (PyTorch Diffusers Engine)")
    st.markdown("Generates a systematic matrix sweeping **4 step values** against **3 guidance scales** with a **fixed seed**, displaying exact visual convergence and CLIP alignment scores.")

    col_g_cfg, col_g_out = st.columns([1, 2.5], gap="medium")
    
    with col_g_cfg:
        st.markdown("#### Grid Settings")
        grid_prompt = st.text_area(
            "Matrix Prompt",
            value="cinematic medium shot of a cyberpunk samurai standing in rain-slicked Tokyo street, glowing neon blue katana, reflective puddle ripples, 8k resolution",
            height=100
        )
        grid_neg = st.text_input("Negative Prompt", value="blurry, distorted, low quality, bad anatomy")
        grid_seed = st.number_input("Fixed Seed (Isolates Parameter Effects)", min_value=0, max_value=999999, value=42)
        
        st.markdown("##### Step Values (Rows = 4)")
        step_1 = st.number_input("Step 1 (Early)", value=5, min_value=1)
        step_2 = st.number_input("Step 2 (Mid-Low)", value=15, min_value=1)
        step_3 = st.number_input("Step 3 (Optimal)", value=25, min_value=1)
        step_4 = st.number_input("Step 4 (Refined)", value=50, min_value=1)
        steps_custom = [step_1, step_2, step_3, step_4]
        
        st.markdown("##### Guidance Scale (Cols = 3)")
        cfg_1 = st.number_input("CFG 1 (Low)", value=1.5, step=0.5)
        cfg_2 = st.number_input("CFG 2 (Standard)", value=7.5, step=0.5)
        cfg_3 = st.number_input("CFG 3 (High/Oversat)", value=15.0, step=0.5)
        guidance_custom = [cfg_1, cfg_2, cfg_3]
        
        run_grid_btn = st.button("🔨 Build 4x3 Comparison Grid", type="primary", use_container_width=True)

    with col_g_out:
        if run_grid_btn or "grid_image" not in st.session_state:
            with st.spinner("Generating 4x3 Parameter Matrix (12 images + CLIP scoring)..."):
                grid_engine = ParameterGridGenerator(
                    pipeline_mgr=st.session_state.pipeline_mgr,
                    clip_scorer=st.session_state.clip_scorer
                )
                
                grid_img, grid_meta = grid_engine.generate_grid(
                    prompt=grid_prompt,
                    negative_prompt=grid_neg,
                    steps_list=steps_custom,
                    guidance_list=guidance_custom,
                    seed=grid_seed,
                    cell_size=256
                )
                st.session_state.grid_image = grid_img
                st.session_state.grid_meta = grid_meta
                
                # Also save to disk deliverable location
                grid_img.save("output/grid_4x3_showcase.png")
                
        st.image(st.session_state.grid_image, caption="Deliverable 2: 4 Steps x 3 Guidance Parameter Grid with Annotated CLIP Alignment Scores", use_container_width=True)
        
        # Download button
        buf = io.BytesIO()
        st.session_state.grid_image.save(buf, format="PNG")
        st.download_button(
            label="💾 Download 4x3 Grid Matrix Image (PNG)",
            data=buf.getvalue(),
            file_name="diffusion_grid_4x3_matrix.png",
            mime="image/png"
        )
        
        # Display CLIP Heatmap table
        st.markdown("#### CLIP Alignment Heatmap (Steps vs Guidance)")
        m_data = st.session_state.grid_meta["matrix"]
        heatmap_rows = []
        for r_idx, s in enumerate(steps_custom):
            row_dict = {"Steps": f"Steps = {s}"}
            for c_idx, g in enumerate(guidance_custom):
                row_dict[f"CFG {g}"] = f"{m_data[r_idx][c_idx]['clip_score']:.2f}"
            heatmap_rows.append(row_dict)
            
        st.dataframe(pd.DataFrame(heatmap_rows).set_index("Steps"), use_container_width=True)

# ==============================================================================
# TAB 3: DCGAN vs DIFFUSION COMPARISON (Deliverable 3 & U2)
# ==============================================================================
with tabs[2]:
    st.subheader("Generative Architecture Showdown: DCGAN vs Diffusion")
    st.markdown("Examines the foundational differences between single-step adversarial mapping ($G(z)$) and iterative score-based denoising trajectories.")
    
    col_comp_act, col_comp_vis = st.columns([1, 2], gap="large")
    
    with col_comp_act:
        st.markdown("#### Comparison Controls")
        comp_seed = st.number_input("Comparison Seed", min_value=0, max_value=999999, value=42)
        comp_prompt = st.text_area(
            "Reference Prompt for CLIP Scoring",
            value="photorealistic studio portrait of an expressive person, 85mm lens, natural lighting",
            height=90
        )
        
        recompare_btn = st.button("⚡ Run Live GAN vs Diffusion Benchmark", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Key Technical Divergences")
        st.markdown("""
        - **Sampling Path**: DCGAN maps $z \\sim \\mathcal{N}(0, I)$ directly to $x$ in **1 single forward pass**. Diffusion denoises iteratively over $T$ steps along reverse stochastic trajectories.
        - **Training Loss**: DCGAN uses adversarial minimax game $\\min_G \\max_D V(D, G)$ which suffers from mode collapse. Diffusion optimizes MSE score matching $\\mathbb{E}[\\|\\epsilon - \\epsilon_\\theta(x_t, t)\\|^2]$ with stable monotonic convergence.
        - **Artifacts**: DCGAN displays transposed convolution checkerboards and facial asymmetry. Diffusion yields photorealistic coherence guided by cross-attention.
        """)

    with col_comp_vis:
        if recompare_btn or "gan_diff_ran" not in st.session_state:
            with st.spinner("Synthesizing DCGAN and Diffusion samples..."):
                # 1. DCGAN Samples
                dcgan_imgs = st.session_state.dcgan_gen.generate_samples(num_samples=4, seed=comp_seed)
                # 2. Diffusion Samples
                diff_imgs = []
                for i in range(4):
                    d_img, _ = st.session_state.pipeline_mgr.generate(
                        prompt=comp_prompt,
                        num_inference_steps=25,
                        guidance_scale=7.5,
                        seed=comp_seed + i,
                        width=256,
                        height=256
                    )
                    diff_imgs.append(d_img)
                    
                # 3. CLIP Scores
                dcgan_clip = st.session_state.clip_scorer.score(dcgan_imgs, comp_prompt)
                diff_clip = st.session_state.clip_scorer.score(diff_imgs, comp_prompt)
                
                st.session_state.dcgan_imgs = dcgan_imgs
                st.session_state.diff_imgs = diff_imgs
                st.session_state.dcgan_clip = dcgan_clip
                st.session_state.diff_clip = diff_clip
                st.session_state.gan_diff_ran = True

        st.markdown("##### 1. DCGAN Output Samples (Radford et al. 2015)")
        st.caption(f"Mean CLIP Alignment: **{np.mean(st.session_state.dcgan_clip):.2f}** | Latency: **~10ms** (1-step forward pass)")
        cols_gan = st.columns(4)
        for idx, col in enumerate(cols_gan):
            with col:
                st.image(st.session_state.dcgan_imgs[idx], caption=f"Sample {idx+1} | CLIP: {st.session_state.dcgan_clip[idx]:.1f}")
                
        st.markdown("##### 2. Diffusion Output Samples (Iterative Reverse Denoising)")
        st.caption(f"Mean CLIP Alignment: **{np.mean(st.session_state.diff_clip):.2f}** | Steps: 25 | CFG: 7.5")
        cols_diff = st.columns(4)
        for idx, col in enumerate(cols_diff):
            with col:
                st.image(st.session_state.diff_imgs[idx], caption=f"Sample {idx+1} | CLIP: {st.session_state.diff_clip[idx]:.1f}")

    # Exhaustive Technical Comparison Matrix
    st.markdown("---")
    st.markdown("#### Comprehensive Architectural & Algorithmic Comparison")
    comp_table = get_gan_vs_diffusion_comparison_table()
    df_comp = pd.DataFrame(comp_table)
    st.table(df_comp.rename(columns={
        "dimension": "Dimension",
        "dcgan": "DCGAN (Radford et al.)",
        "diffusion": "Diffusion (Latent / DDPM)",
        "advantage": "Key Advantage"
    }))

# ==============================================================================
# TAB 4: NOISE AS PROBABILITY DISTRIBUTION & SCHEDULES (U1)
# ==============================================================================
with tabs[3]:
    st.subheader("Unit 1: Noise as a Probability Distribution & SDE Dynamics")
    
    col_u1_math, col_u1_sched = st.columns([1, 1], gap="large")
    
    with col_u1_math:
        st.markdown("#### 1. Standard Normal Distribution $\\mathcal{N}(0, I)$ in Latent Space")
        st.markdown("""
        In Latent Diffusion, the initial state $z_T$ is sampled from an isotropic Gaussian distribution:
        $$p(z) = \\frac{1}{(2\\pi)^{D/2}} \\exp\\left(-\\frac{1}{2}\\|z\\|^2\\right)$$
        For SD 512x512, latent dimensionality is $4 \\times 64 \\times 64 = 16,384$ dimensions.
        """)
        
        sample_size = st.slider("Empirical Sample Size (Pixels / Latents)", 1000, 50000, 16384, step=1000)
        noise_arr = np.random.normal(0, 1, size=(sample_size,)).astype(np.float32)
        metrics_dict = compute_distribution_metrics(noise_arr)
        
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Empirical Mean", f"{metrics_dict['mean']:.4f}")
        m_c2.metric("Variance", f"{metrics_dict['variance']:.4f}")
        m_c3.metric("Skewness", f"{metrics_dict['skewness']:.4f}")
        m_c4.metric("Excess Kurtosis", f"{metrics_dict['excess_kurtosis']:.4f}")
        
        # Plot Gaussian Histogram vs Theoretical PDF
        fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0f172a")
        ax.set_facecolor("#1e293b")
        count, bins, _ = ax.hist(noise_arr, bins=60, density=True, alpha=0.6, color="#60a5fa", edgecolor="#3b82f6")
        # Theoretical Bell Curve
        pdf = 1.0 / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * bins**2)
        ax.plot(bins, pdf, color="#f43f5e", linewidth=2.5, label="Theoretical PDF N(0, 1)")
        ax.set_title("Empirical Distribution vs Gaussian PDF", color="white", fontsize=11)
        ax.tick_params(colors="#94a3b8")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig)

    with col_u1_sched:
        st.markdown("#### 2. Forward Variance Schedules: $\\beta_t$, $\\alpha_t$, $\\bar{\\alpha}_t$")
        sched_type = st.selectbox("Schedule Type", ["scaled_linear", "linear", "cosine"], index=0)
        num_t = st.slider("Total Timesteps (T)", min_value=100, max_value=1000, value=1000, step=100)
        
        sched_data = get_beta_schedule(schedule_type=sched_type, num_timesteps=num_t)
        
        fig2, ax2 = plt.subplots(figsize=(6, 3.2), facecolor="#0f172a")
        ax2.set_facecolor("#1e293b")
        t_axis = np.arange(num_t)
        ax2.plot(t_axis, sched_data["sqrt_alphas_cumprod"], color="#34d399", label="Signal factor $\\sqrt{\\bar{\\alpha}_t}$", linewidth=2)
        ax2.plot(t_axis, sched_data["sqrt_one_minus_alphas_cumprod"], color="#f87171", label="Noise factor $\\sqrt{1 - \\bar{\\alpha}_t}$", linewidth=2)
        ax2.set_title(f"Forward Diffusion Variance Dynamics ({sched_type})", color="white", fontsize=11)
        ax2.set_xlabel("Timestep t", color="#94a3b8")
        ax2.tick_params(colors="#94a3b8")
        ax2.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
        for spine in ax2.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig2)
        
        st.caption("Signal-to-Noise Ratio: $\\text{SNR}(t) = \\frac{\\bar{\\alpha}_t}{1 - \\bar{\\alpha}_t}$. As $t \\to T$, signal vanishes and state approaches pure Gaussian noise.")

    st.markdown("---")
    st.markdown("#### 3. Interactive Forward Noising Simulation: $q(x_t | x_0)$")
    t_slider = st.slider("Select Timestep t", 0, num_t - 1, value=300)
    
    # Generate test image
    size = 256
    y, x = np.mgrid[0:size, 0:size]
    circle = np.sqrt((x - size/2)**2 + (y - size/2)**2) < (size * 0.35)
    test_img = np.zeros((size, size, 3), dtype=np.float32)
    test_img[circle] = [0.2, 0.6, 0.9]
    test_img[~circle] = [0.85, 0.4, 0.2]
    x_0 = (test_img * 2.0 - 1.0).astype(np.float32)
    
    sqrt_ab = sched_data["sqrt_alphas_cumprod"][t_slider]
    sqrt_1_ab = sched_data["sqrt_one_minus_alphas_cumprod"][t_slider]
    eps = np.random.RandomState(42).normal(0, 1, size=x_0.shape).astype(np.float32)
    x_t = sqrt_ab * x_0 + sqrt_1_ab * eps
    
    col_x0, col_eps, col_xt = st.columns(3)
    with col_x0:
        st.image((x_0 + 1.0) / 2.0, caption="Initial Signal $x_0$ (t = 0)")
    with col_eps:
        st.image((eps - eps.min()) / (eps.max() - eps.min()), caption="Sampled Gaussian Noise $\\epsilon \\sim \\mathcal{N}(0, I)$")
    with col_xt:
        st.image(np.clip((x_t + 1.0) / 2.0, 0, 1), caption=f"Noised State $x_{{{t_slider}}} = {sqrt_ab:.2f} x_0 + {sqrt_1_ab:.2f} \\epsilon$")

# ==============================================================================
# TAB 5: ETHICS, SAFETY & REPRESENTATION (U5)
# ==============================================================================
with tabs[4]:
    st.subheader("Unit 5: NSFW Safety Filters, Model Licensing & Representation Audit")
    
    col_u5_left, col_u5_right = st.columns([1, 1], gap="large")
    
    with col_u5_left:
        st.markdown("#### 1. NSFW Safety Filter Verification")
        st.markdown("""
        The Stable Diffusion safety checker extracts CLIP visual embeddings and evaluates cosine similarity against sensitive concept hyperplanes:
        $$\\text{Flagged} = \\mathbb{I}\\left(\\cos(\\mathbf{v}_{\\text{image}}, \\mathbf{c}_{\\text{banned}}) > \\tau\\right)$$
        If flagged, the image is suppressed and substituted with an opaque placeholder.
        """)
        
        test_safety_prompt = st.text_input("Test Safety Evaluation Prompt", value="A peaceful botanical garden at sunset")
        safety_inst = SafetyChecker(enabled=True)
        
        if st.button("Inspect Safety Classifier"):
            sample_clean = Image.new("RGB", (256, 256), (34, 197, 94))
            _, is_flagged, meta_s = safety_inst.check_image(sample_clean, test_safety_prompt)
            st.json(meta_s)
            
        st.markdown("---")
        st.markdown("#### 2. CreativeML Open RAIL-M License Audit")
        lic_data = get_open_rail_m_license_summary()
        st.markdown(f"**License**: `{lic_data['license_name']}`")
        st.markdown("**Permissible Rights:**")
        for freedom in lic_data["core_freedoms"]:
            st.markdown(f"- ✅ {freedom}")
        st.markdown("**Attachment A Prohibited Use Cases:**")
        for prohib in lic_data["attachment_a_prohibited_uses"][:4]:
            st.markdown(f"- 🚫 {prohib}")

    with col_u5_right:
        st.markdown("#### 3. Representation & Demographic Bias Audit (LAION-5B)")
        st.markdown("""
        Web-scale pre-training data manifests pronounced societal skews:
        - **Occupational Stereotyping**: Prompts without gender qualification inherit historical occupational skews.
        - **Western Centricity**: Default backgrounds favor Western architecture, clothing, and rituals.
        """)
        
        rep_audit = get_representation_and_bias_audit()
        for b_item in rep_audit["documented_biases"]:
            with st.expander(f"📌 {b_item['category']}", expanded=True):
                st.markdown(f"**Finding**: {b_item['finding']}")
                st.markdown(f"**Engineering Mitigation**: `{b_item['mitigation']}`")
                
        st.markdown("##### Benchmark Evaluation Prompts")
        df_bench = pd.DataFrame(rep_audit["benchmark_prompts_to_test"])
        st.dataframe(df_bench, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>P33 · Diffusion Parameter Explorer | PyTorch, Diffusers, CLIP & Streamlit</div>", unsafe_allow_html=True)

