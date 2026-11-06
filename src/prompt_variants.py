"""
Prompt Variants and Prompt Engineering Engine for Diffusion Models
Unit 1 Deliverable component.
"""

from typing import Dict, List, Optional
import dataclasses

@dataclasses.dataclass
class PromptVariant:
    category: str
    name: str
    positive_prompt: str
    negative_prompt: str
    recommended_steps: int
    recommended_guidance: float
    description: str
    rationale: str

# Curated Prompt Variants library showcasing different stylistic, technical, and compositional dimensions
CURATED_PROMPTS: Dict[str, List[PromptVariant]] = {
    "Cyberpunk & Sci-Fi": [
        PromptVariant(
            category="Cyberpunk & Sci-Fi",
            name="Cinematic Cyberpunk Samurai",
            positive_prompt="cinematic medium shot of a cyberpunk samurai standing in rain-slicked Tokyo street, glowing neon blue katana, reflective puddle ripples, hyper-detailed armor with carbon fiber textures, volumetrical atmospheric fog, 8k resolution, photorealistic, 35mm lens, f/1.8",
            negative_prompt="blurry, deformed limbs, bad anatomy, lowres, flat lighting, cartoonish, plastic texture, text, watermark, mutated fingers, out of focus",
            recommended_steps=25,
            recommended_guidance=7.5,
            description="High contrast, sharp reflections, neon lighting with deep volumetric shadows.",
            rationale="Tests cross-attention sensitivity to lighting words ('glowing neon', 'volumetric fog') and high-frequency reflections."
        ),
        PromptVariant(
            category="Cyberpunk & Sci-Fi",
            name="Futuristic Botanical Greenhouse",
            positive_prompt="interior view of an orbital space greenhouse, bioluminescent alien flora, glass biodome revealing planet earth in background, soft morning sunlight, architectural realism, clean geometry, 8k octane render",
            negative_prompt="blurry, dark, noisy, low quality, oversaturated, ugly, deformed, grain, text",
            recommended_steps=30,
            recommended_guidance=7.0,
            description="Intricate biological structures juxtaposed with clean futuristic glass architecture.",
            rationale="Tests how diffusion handles intricate natural forms (veins of leaves) alongside geometric glass panes."
        )
    ],
    "Photorealistic Portraits": [
        PromptVariant(
            category="Photorealistic Portraits",
            name="Studio Portrait of an Old Craftsman",
            positive_prompt="close-up portrait of an elderly watchmaker with deep smile lines and gray beard, brass magnifying loupe attached to glasses, holding fine tweezers, dramatic Rembrandt studio lighting, rich texture of skin pores, 85mm portrait lens, award-winning photography",
            negative_prompt="doll skin, airbrushed, plastic face, blur, extra eyes, bad anatomy, poorly drawn hands, duplicate, low resolution",
            recommended_steps=35,
            recommended_guidance=8.0,
            description="Fine skin pores, wrinkles, and metallic watch components.",
            rationale="Demonstrates how high guidance scale pulls realistic textures without blowing out dynamic range."
        ),
        PromptVariant(
            category="Photorealistic Portraits",
            name="Wildlife Macro - Tree Frog",
            positive_prompt="extreme macro photograph of a vibrant red-eyed tree frog perched on a rainforest leaf with water droplets, soft bokeh background, glistening moisture texture, National Geographic style, sharp focus on eyes",
            negative_prompt="cartoon, digital drawing, low quality, out of frame, blurry, deformed limbs, dull colors",
            recommended_steps=30,
            recommended_guidance=7.5,
            description="Organic water droplets, high refractive index, and vivid amphibian skin texture.",
            rationale="Highlights fine-scale surface scattering and focal falloff (depth of field)."
        )
    ],
    "Artistic & Stylized": [
        PromptVariant(
            category="Artistic & Stylized",
            name="Impressionist Mountain Lake",
            positive_prompt="oil painting of alpine mountain reflection in crystal lake, thick impasto palette knife textures, warm sunset glow, dynamic brush strokes, inspired by Claude Monet and John Singer Sargent, rich painterly colors",
            negative_prompt="photograph, smooth digital rendering, 3d render, plastic, modern, watermark, blurry",
            recommended_steps=25,
            recommended_guidance=7.0,
            description="Thick paint texture, palette knife strokes, and soft romantic lighting.",
            rationale="Illustrates how diffusion captures artistic medium medium characteristics (brush marks, canvas grain) rather than photographic pixels."
        ),
        PromptVariant(
            category="Artistic & Stylized",
            name="Studio Ghibli Meadow",
            positive_prompt="peaceful grassy hillside with rolling green meadows, fluffy billowing clouds in deep azure sky, quaint wooden cottage, pastel color grading, Studio Ghibli anime background art, Hayao Miyazaki style, hand drawn feel, 4k",
            negative_prompt="photorealistic, hyperrealistic, dark, gritty, horror, oversaturated neon, 3d cgi render, noisy",
            recommended_steps=20,
            recommended_guidance=6.5,
            description="Soft hand-painted anime aesthetic with lush pastel grass and puffy cumulus clouds.",
            rationale="Demonstrates stylized cel shading and flat tonal transitions with moderate guidance."
        )
    ],
    "Architecture & Environment": [
        PromptVariant(
            category="Architecture & Environment",
            name="Minimalist Modern Villa in Forest",
            positive_prompt="architectural photography of a minimalist cantilevered concrete villa nestled among pine forest, floor-to-ceiling glass walls, warm interior lighting during twilight, tranquil reflection pool, architectural digest, brutalist elegance",
            negative_prompt="messy, cluttered, distorted geometry, bent walls, lowres, oversaturated, blurry, noise",
            recommended_steps=30,
            recommended_guidance=8.0,
            description="Crisp straight lines, concrete textures, and architectural twilight illumination.",
            rationale="Tests diffusion's spatial consistency on straight lines, perspective vanishing points, and material contrast."
        )
    ]
}

def generate_prompt_variant(
    subject: str,
    medium_style: str = "photorealistic",
    lighting: str = "cinematic lighting",
    camera_view: str = "medium shot",
    quality_boosters: Optional[List[str]] = None,
    mood: str = "dramatic"
) -> Dict[str, str]:
    """
    Systematic prompt engineering builder combining modular semantic components.
    Returns positive and negative prompts tailored to diffusion models.
    """
    if quality_boosters is None:
        quality_boosters = ["highly detailed", "8k resolution", "sharp focus"]
        
    style_presets = {
        "photorealistic": "35mm photograph, natural skin textures, realistic depth of field, f/2.8",
        "cinematic": "cinematic still, Panavision anamorphic lens, color graded, film grain",
        "digital art": "digital concept art, Artstation trending, intricate details, octane render",
        "oil painting": "oil on canvas, rich impasto, visible brushstrokes, classical masterpiece",
        "watercolor": "delicate watercolor washes, wet-on-wet technique, textured cold press paper",
        "anime": "anime key visual, clean lineart, vibrant anime aesthetic, Studio Ghibli vibe",
        "isometric 3d": "isometric 3D render, miniature diorama, cute proportions, ambient occlusion"
    }
    
    negative_presets = {
        "photorealistic": "cartoon, illustration, painting, plastic skin, distorted, blurry, extra limbs, bad anatomy, watermark",
        "cinematic": "amateur photo, washed out, low contrast, oversaturated, watermark, bad framing, deformed",
        "digital art": "photograph, amateur sketch, blurry, compression artifacts, flat colors, watermark",
        "oil painting": "photograph, 3d render, vector graphic, digital smoothness, bad anatomy",
        "watercolor": "heavy oil paint, dark digital renders, photography, muddy colors, harsh lines",
        "anime": "photorealistic, hyperrealistic, western comic style, bad anatomy, extra fingers",
        "isometric 3d": "flat 2d, sketch, realistic photo, dirty texture, chaotic background"
    }
    
    style_desc = style_presets.get(medium_style.lower(), medium_style)
    neg_desc = negative_presets.get(medium_style.lower(), "blurry, low quality, deformed, artifacts, watermark")
    
    positive = f"{camera_view} of {subject}, {style_desc}, {lighting}, {mood} atmosphere, {', '.join(quality_boosters)}"
    negative = f"{neg_desc}, deformed limbs, bad anatomy, duplicate, lowres, text, signature"
    
    return {
        "positive": positive,
        "negative": negative,
        "subject": subject,
        "style": medium_style
    }

