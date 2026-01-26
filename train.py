import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_components import UNetModel
from torch.optim.adamw import AdamW
from torch.nn.functional import mse_loss
import utils
import os
from tqdm import tqdm
import torch._inductor
torch._inductor.config.layout_optimization=False #type: ignore

device="cuda"

IMAGE_DIM = 256
UNET_DTYPE = torch.bfloat16
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2
LOG_STEPS = 64

ckpt_dir = "./checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

mnist_transform = transforms.Compose([
    transforms.Resize(IMAGE_DIM),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=mnist_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE // GRAD_ACCUM_STEPS,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

def train_unet(train_loader):
    vae = utils.VAEManager(0, raw_input_channels=1)
    text_encoder = utils.TextEncoderManager(63)
    vae_output_channels = vae.channels
    vae_downsampling = vae.downsampling
    d_text = text_encoder.output_dim
    num_unet_layers = 10

    assert num_unet_layers % 2 == 0
    assert IMAGE_DIM % vae_downsampling == 0
    assert 2 ** (num_unet_layers // 2) <= IMAGE_DIM // vae_downsampling

    unet = UNetModel(
        num_layers=3, # 6 total layers, 3 up and down each
        raw_input_channels=vae_output_channels,
        hidden_size_channels=16,
        d_text=d_text,
        num_heads=4,
        num_kv_heads=2,
        fan_out_factor=4
    ).to(device, dtype=torch.bfloat16)
    unet.train()
    optim = AdamW(unet.parameters(), lr=3e-5, betas=(0.9, 0.999), weight_decay=1e-2, fused=True)
    unet = torch.compile(unet)
    optim.zero_grad()
    global_step = 1
    for epoch in range(1, 50):
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            dynamic_ncols=True,
            desc="Training UNet"
        )
        for step, (image_tensors, image_classes) in progress_bar:
            prompts = utils.generate_prompts(image_classes)
            encoded_prompts = text_encoder.encode(prompts)
            noisy_latents, noise, timesteps = vae.apply_gaussian_noise_and_encode(image_tensors, 1000)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = unet(noisy_latents, timesteps, encoded_prompts)
                loss = mse_loss(preds, noise) / GRAD_ACCUM_STEPS
            loss.backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=0.5) #type: ignore
                optim.step()
                optim.zero_grad()
                global_step += 1
                if global_step % LOG_STEPS == 0:
                    progress_bar.set_postfix(
                        loss=f"{(loss * GRAD_ACCUM_STEPS):.4f}",
                        step=global_step
                    )
                if global_step % (LOG_STEPS * 4) == 0:
                    ckpt_path = os.path.join(
                        ckpt_dir,
                        f"unet_step{global_step}.pt"
                    )
                    torch.save(
                        {
                            "step": global_step,
                            "model_state_dict": (unet._orig_mod if hasattr(unet, "_orig_mod") else unet).state_dict(), # type: ignore
                            "optimizer_state_dict": optim.state_dict(),
                            "loss": loss.item(),
                        },
                        ckpt_path
                    )

def train_pixel_dit(train_loader):
    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        dynamic_ncols=True,
        desc="Training UNet"
    )
    text_encoder = utils.TextEncoderManager(63)
    d_text = text_encoder.output_dim


    

train_unet(train_loader)