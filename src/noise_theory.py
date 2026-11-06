"""
Unit 1: Noise as a Probability Distribution & Diffusion Schedule Dynamics.
Provides mathematical formulations, empirical distributions, schedule simulations,
and forward/reverse diffusion math.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from PIL import Image

def get_beta_schedule(
    schedule_type: str = "scaled_linear",
    num_timesteps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.0120
) -> Dict[str, np.ndarray]:
    """
    Computes beta, alpha, and cumulative alpha_bar schedules.
    
    Types:
      - 'linear': Standard DDPM linear schedule (Ho et al.)
      - 'cosine': Cosine schedule (Nichol & Dhariwal, 2021)
      - 'scaled_linear': Scaled linear schedule used in Latent Diffusion / Stable Diffusion
    """
    if schedule_type == "linear":
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    elif schedule_type == "scaled_linear":
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=np.float64) ** 2
    elif schedule_type == "cosine":
        s = 0.008
        steps = num_timesteps + 1
        t = np.linspace(0, num_timesteps, steps, dtype=np.float64) / num_timesteps
        alphas_bar = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = np.clip(betas, a_min=1e-5, a_max=0.999)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    
    # Forward diffusion parameters: q(x_t | x_0) = N(sqrt(alpha_bar_t)*x_0, (1 - alpha_bar_t)*I)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    
    # Signal-to-noise ratio: SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
    snr = alphas_cumprod / (1.0 - alphas_cumprod + 1e-8)
    log_snr = np.log(np.maximum(snr, 1e-12))
    
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "snr": snr,
        "log_snr": log_snr
    }

def sample_gaussian_noise(
    shape: Tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates standard multivariate Gaussian noise z ~ N(mu, sigma^2 * I).
    Diffusion models initialize from this prior distribution at t = T.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    noise = rng.normal(loc=mean, scale=std, size=shape).astype(np.float32)
    return noise

def forward_diffusion_step(
    x_0: np.ndarray,
    t: int,
    noise: Optional[np.ndarray] = None,
    schedule: Optional[Dict[str, np.ndarray]] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-form sampling for forward diffusion:
      x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    where epsilon ~ N(0, I).
    
    Returns:
      (x_t, epsilon)
    """
    if schedule is None:
        schedule = get_beta_schedule()
        
    sqrt_alpha_bar = schedule["sqrt_alphas_cumprod"][t]
    sqrt_one_minus_alpha_bar = schedule["sqrt_one_minus_alphas_cumprod"][t]
    
    if noise is None:
        noise = sample_gaussian_noise(x_0.shape, seed=seed)
        
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise

def simulate_diffusion_trajectory(
    image_np: np.ndarray,
    timesteps: List[int],
    schedule_type: str = "scaled_linear",
    seed: int = 42
) -> List[Tuple[int, np.ndarray]]:
    """
    Simulates the forward noising trajectory over specified timesteps.
    image_np should be normalized to [-1, 1].
    """
    schedule = get_beta_schedule(schedule_type=schedule_type)
    rng = np.random.RandomState(seed)
    base_noise = rng.normal(0, 1, size=image_np.shape).astype(np.float32)
    
    trajectory = []
    for t in timesteps:
        t_clamped = min(max(t, 0), len(schedule["betas"]) - 1)
        x_t, _ = forward_diffusion_step(image_np, t_clamped, noise=base_noise, schedule=schedule)
        trajectory.append((t_clamped, x_t))
    return trajectory

def compute_distribution_metrics(data: np.ndarray) -> Dict[str, float]:
    """
    Empirical distribution moments to prove Gaussian distribution properties:
    Mean, Std, Skewness (should be ~0 for normal), Excess Kurtosis (should be ~0 for normal).
    """
    flat = data.flatten()
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    
    # Standardized moments
    m2 = np.mean((flat - mean) ** 2)
    m3 = np.mean((flat - mean) ** 3)
    m4 = np.mean((flat - mean) ** 4)
    
    skewness = float(m3 / (m2 ** 1.5 + 1e-8))
    kurtosis = float(m4 / (m2 ** 2 + 1e-8) - 3.0)  # Excess kurtosis
    
    return {
        "mean": mean,
        "std": std,
        "variance": float(std ** 2),
        "skewness": skewness,
        "excess_kurtosis": kurtosis,
        "min": float(np.min(flat)),
        "max": float(np.max(flat))
    }

