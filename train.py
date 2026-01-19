import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_components import UNetModel
from torch.optim.adamw import AdamW
from torch.nn.functional import mse_loss
import utils
import os
from tqdm import tqdm

device="cuda"

IMAGE_DIM = 128
UNET_DTYPE = torch.float16
BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 4
LOG_STEPS = 100

ckpt_dir = "./checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

mnist_transform = transforms.Compose([
    transforms.Resize(IMAGE_DIM),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=3),
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
    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        dynamic_ncols=True,
        desc="Training UNet"
    )
    vae = utils.VAEManager(83)
    text_encoder = utils.TextEncoderManager(63)
    vae_output_channels = vae.channels
    vae_downsampling = vae.downsampling
    d_text = text_encoder.output_dim
    num_unet_layers = 8

    assert num_unet_layers % 2 == 0
    assert IMAGE_DIM % vae_downsampling == 0
    assert 2 ** (num_unet_layers // 2) <= IMAGE_DIM // vae_downsampling

    unet = UNetModel(
        num_layers=num_unet_layers // 4,
        input_channels=vae_output_channels,
        d_text=d_text,
        num_heads=2,
        num_kv_heads=1
    ).to(device, dtype=torch.float16)
    unet.train()
    optim = AdamW(unet.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2)
    unet = torch.compile(unet)
    optim.zero_grad()
    for step, (image_tensors, image_classes) in progress_bar:
        prompts = utils.generate_prompts(image_classes)
        encoded_prompts = text_encoder.encode(prompts)
        noisy_latents, noise, timesteps = vae.apply_gaussian_noise_and_encode(image_tensors, 500)
        preds = unet(noisy_latents, timesteps, encoded_prompts)
        loss = mse_loss(preds, noise)
        loss.backward()
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0) #type: ignore
            optim.step()
            optim.zero_grad()
            global_step = (step+1) // GRAD_ACCUM_STEPS
            if global_step % LOG_STEPS == 0:
                loss_val = loss.item()
                progress_bar.set_postfix(
                    loss=f"{loss:.4f}",
                    step=global_step
                )

                ckpt_path = os.path.join(
                    ckpt_dir,
                    f"unet_step{global_step}.pt"
                )
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": (unet._orig_mod if hasattr(unet, "_orig_mod") else unet).state_dict(), # type: ignore
                        "optimizer_state_dict": optim.state_dict(),
                        "loss": loss_val,
                    },
                    ckpt_path
                )

train_unet(train_loader)