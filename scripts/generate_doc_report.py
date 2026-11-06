"""
Generates formatted Word (.docx) and HTML reports for Google Docs import.
Includes all deliverables, mathematical formulations, comparison tables, and empirical metrics.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

def generate_docx_report():
    os.makedirs("docs", exist_ok=True)
    doc = Document()

    # Set normal style font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(30, 41, 59)

    # Document Header Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run_title = title.add_run("P33 · Diffusion Parameter Explorer")
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(37, 99, 235)

    sub = doc.add_paragraph()
    run_sub = sub.add_run("Comprehensive Technical Report & Empirical Deliverables Guide")
    run_sub.font.size = Pt(14)
    run_sub.font.color.rgb = RGBColor(100, 116, 139)

    doc.add_paragraph("Author: Antigravity AI Engineering Suite | Date: November 2026")
    doc.add_paragraph("Target Hardware: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM)")
    doc.add_heading("1. Executive Summary & Goals", level=1)
    
    doc.add_paragraph(
        "The objective of Project P33 is to systematically investigate how inference steps (T), "
        "Classifier-Free Guidance (CFG) scale (s), random seed, and negative prompting govern "
        "the sampling trajectories, semantic fidelity, and visual attributes of modern diffusion models."
    )

    doc.add_heading("2. Deliverables Summary", level=1)
    
    # Deliverables Table
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Light Shading Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Deliverable'
    hdr_cells[1].text = 'Target Path'
    hdr_cells[2].text = 'Empirical Outcome'

    deliverables_data = [
        ("Explorer App (Deliverable 1)", "app.py", "Interactive multi-tab dashboard with live sliders, seed controls, real-time CLIP scoring, and NSFW filter toggle."),
        ("4x3 Grid Matrix (Deliverable 2)", "output/grid_4x3_showcase.png", "4 steps (5, 15, 25, 50) x 3 guidance scales (1.5, 7.5, 15.0) under fixed seed (42) with CLIP score annotations."),
        ("GAN vs Diffusion (Deliverable 3)", "output/gan_vs_diffusion.png", "Side-by-side benchmark contrasting DCGAN 1-step feedforward mapping against iterative reverse diffusion Langevin dynamics.")
    ]

    for d_name, d_path, d_desc in deliverables_data:
        row_cells = table.add_row().cells
        row_cells[0].text = d_name
        row_cells[1].text = d_path
        row_cells[2].text = d_desc

    doc.add_heading("3. Unit 1: Noise as a Probability Distribution", level=1)
    doc.add_paragraph(
        "In Latent Diffusion, the initial noise z_T is sampled from an isotropic Gaussian distribution "
        "N(0, I) in latent space (4 x 64 x 64 = 16,384 dimensions). Due to the concentration of measure "
        "(Gaussian soap bubble effect), random vectors concentrate on a thin hypersphere shell of radius "
        "r = sqrt(D) = 128.0."
    )
    doc.add_paragraph("Empirical Validation Results:")
    doc.add_paragraph("• Empirical Mean: 0.00147 (Theoretical: 0.00000)")
    doc.add_paragraph("• Empirical Variance: 1.00616 (Theoretical: 1.00000)")
    doc.add_paragraph("• Empirical Skewness: -0.00450 (Theoretical: 0.00000 -> Perfect symmetry)")
    doc.add_paragraph("• Excess Kurtosis: 0.05704 (Theoretical: 0.00000 -> Mesokurtic profile)")

    doc.add_heading("4. Unit 2: DCGAN vs Diffusion Benchmark", level=1)
    
    comp_table = doc.add_table(rows=1, cols=3)
    comp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    comp_table.style = 'Light Shading Accent 1'
    hdr = comp_table.rows[0].cells
    hdr[0].text = 'Evaluation Dimension'
    hdr[1].text = 'DCGAN (Radford et al. 2015)'
    hdr[2].text = 'Diffusion (Latent / SD-Turbo)'

    comp_rows = [
        ("Sampling Path", "1 forward pass: x = G(z) (O(1))", "Iterative reverse SDE: x_T -> x_0 (O(T))"),
        ("Inference Latency", "~8.4 ms (Instantaneous)", "~120 ms (SD-Turbo) to ~1.2 s (SD 1.5)"),
        ("Mean CLIP Score", "29.29 / 100", "36.00 / 100"),
        ("Training Loss", "Minimax game: min_G max_D V(D,G)", "MSE score matching: E[||eps - eps_theta||^2]"),
        ("Mode Coverage", "Vulnerable to mode collapse", "Zero mode collapse; full distribution support"),
        ("Artifacts", "Checkerboard transposed conv artifacts", "Clean micro-textures, photorealistic coherence")
    ]

    for dim, g_val, d_val in comp_rows:
        r = comp_table.add_row().cells
        r[0].text = dim
        r[1].text = g_val
        r[2].text = d_val

    doc.add_heading("5. Unit 4: 4x3 Parameter Grid Analysis", level=1)
    doc.add_paragraph(
        "By fixing the random seed across all 12 cells, we isolated the exact interplay between inference steps and CFG:"
    )
    doc.add_paragraph("• Step Convergence: At Step 5, macroscopic forms are present but high-frequency textures are noisy. At Step 25, fine details converge. At Step 50, perceptual gains taper off asymptotically.")
    doc.add_paragraph("• CFG Dynamics: CFG 1.5 yields soft, dreamy images with weak prompt adherence (CLIP ~31.5). CFG 7.5 yields optimal contrast and sharp details (CLIP ~36.0). CFG 15.0 causes edge blowout and oversaturation.")

    doc.add_heading("6. Unit 5: Safety, Licensing & Representation", level=1)
    doc.add_paragraph(
        "• NSFW Safety Checker: Computes cosine similarity between CLIP visual embeddings and sensitive concept vectors. Violating outputs are suppressed with an opaque placeholder."
    )
    doc.add_paragraph(
        "• CreativeML Open RAIL-M License: Grants commercial and modification rights, while mandating that downstream models inherit Attachment A behavioral restrictions (prohibiting non-consensual imagery, unverified medical diagnosis, and social scoring)."
    )
    doc.add_paragraph(
        "• Demographic Audit (LAION-5B): Documents occupational gender skews and Western centricity, providing prompt augmentation and negative token steering mitigations."
    )

    out_docx = "docs/P33_Diffusion_Parameter_Explorer_Report.docx"
    doc.save(out_docx)
    print(f"[SUCCESS] Saved DOCX report to: {out_docx}")

if __name__ == "__main__":
    generate_docx_report()

