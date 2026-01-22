from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler 
import torch
import torch.nn as nn
import random

# UNet utils

class TextEncoderManager:
    def __init__(self, param_budget_millions, device="cuda", dtype=torch.float16):
        """
        Uses CLIP and takes parameter budget into account when selecting which CLIP model to use.
        Selects the model with the largest number of params below or equal to provided budget.
        Returns output_dims.
        """
        if param_budget_millions < 63:
            raise ValueError("Smallest Text Encoder has 63M params.")
        options = {
            63 : {
                "model" : "openai/clip-vit-base-patch32",
                "output_dim" : 512
            },
            120 : {
                "model" : "openai/clip-vit-base-patch14",
                "output_dim" : 768
            },
            300 : {
                "model" : "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                "output_dim" : 768
            },
            700 : {
                "model" : "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                "output_dim" : 1280
            },
        }
        selected_params = max([params for params in options.keys() if params <= param_budget_millions])
        self.tokenizer = CLIPTokenizer.from_pretrained(options[selected_params]["model"])
        self.encoder = CLIPTextModel.from_pretrained(options[selected_params]["model"], torch_dtype=dtype)
        self.output_dim = options[selected_params]["output_dim"]
        self.model_name = options[selected_params]["model"]
        self.device = device
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(self.device) #type: ignore
    
    def encode(self, batch_text):
        tokens = self.tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        outputs = self.encoder(**tokens)
        return outputs.last_hidden_state


class VAEManager:
    def __init__(self, param_budget_millions, device="cuda", dtype=torch.float16):
        if param_budget_millions < 15:
            raise ValueError("Minimum VAE params is 15M")
        options = {
            15 : {
                "model" : "SG161222/Mobile-VAE-0.4",
                "downsampling": 16,
                "channels": 4,
            },
            25 : {
                "model" : "madebyollin/tiny-vae",
                "downsampling": 8,
                "channels": 4,
            },
            83 : {
                "model" : "stabilityai/sd-vae-ft-ema",
                "downsampling": 8,
                "channels": 4,
            },
            90 : {
                "model" : "stabilityai/sdxl-vae",
                "downsampling": 8,
                "channels": 4,
            },
        }
        selected_params = max([params for params in options.keys() if params <= param_budget_millions])
        self.vae = AutoencoderKL.from_pretrained(options[selected_params]["model"], torch_dtype=dtype)
        self.downsampling = options[selected_params]["downsampling"]
        self.channels = options[selected_params]["channels"]
        self.model_name = options[selected_params]["model"]
        self.device = device
        self.scheduler = DDPMScheduler()
        self.vae = self.vae.to(self.device) #type:ignore
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def apply_gaussian_noise_and_encode(self, batch_image_tensors, max_timestep):
        """
        Assumes input dims of batch, 3 (channels), H, W (RGB and previously resized).
        Returns: {
            noisy_latents : [B, C, H/self.downsampling, W/self.downsampling],
            noise: [B, C, H/self.downsampling, W/self.downsampling],
            timesteps: [B]
        }
        """
        batch_image_tensors = batch_image_tensors.to(self.device, dtype=torch.float16)
        latents = self.vae.encode(batch_image_tensors).latent_dist.sample() * self.vae.config.scaling_factor #type: ignore
        noise = torch.randn_like(latents)
        timesteps = self._sample_random_timesteps(latents.shape[0], max_timestep=min(self.scheduler.config.num_train_timesteps, max_timestep)) #type: ignore
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps) #type: ignore
        return noisy_latents, noise, timesteps

    def _sample_random_timesteps(self, batch_size, max_timestep):
        return torch.randint(
            low=0,
            high=max_timestep,
            size=(batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    def decode(self, latents):
        latents = latents / self.vae.config.scaling_factor #type: ignore
        images = self.vae.decode(latents).sample # type: ignore
        return images

# PixelDiT utils

# General utils

def finite(name, x):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"NON-FINITE IN {name}: min={x.nan_to_num().min().item()}, max={x.nan_to_num().max().item()}")

def generate_prompts(batch_targets):
    prompt_templates = [
        "Generate me an image of {cls}", "Show me a {cls}", "Create a {cls}", "Draw a {cls}", "Make a {cls}",
        "I want to see a {cls}", "Produce a {cls}", "Render a {cls}", "Picture of a {cls}", "Write the number {cls}",
        "Sketch a {cls}", "Can you make a {cls}", "Give me a {cls}", "Display a {cls}", "Visualize a {cls}",
        "A {cls} please", "I need a {cls}", "Design a {cls}", "Print a {cls}", "Output a {cls}",
        "Make the digit {cls}", "Show the number {cls}", "Create the digit {cls}", "Generate a handwritten {cls}", "Draw the number {cls}",
        "I'd like a {cls}", "Produce the digit {cls}", "Can I see a {cls}", "Make me a {cls}", "Write out a {cls}",
        "A picture of the number {cls}", "Generate a {cls} digit", "Show me the digit {cls}", "Create a picture of {cls}", "Draw me a {cls}",
        "I want to see the number {cls}", "Give me the digit {cls}", "Produce an image of {cls}", "Render the number {cls}", "Can you draw a {cls}",
        "Make a picture of {cls}", "I'd like to see a {cls}", "Output the digit {cls}", "Display the number {cls}", "Generate a {cls} for me",
    ]
    return [random.choice(prompt_templates).format(cls=cls) for cls in batch_targets]