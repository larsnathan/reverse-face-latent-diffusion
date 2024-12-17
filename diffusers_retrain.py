from diffusers import UNet2DModel, UNet2DConditionModel, DDIMScheduler, VQModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
import re
import tqdm
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 40
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-retrain-128"  # the model name locally and on the HF Hub

    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

class CelebADataset(Dataset):
    def __init__(self, image_folder, embedding_folder, transform=None):
        self.image_folder = image_folder
        self.embedding_folder = embedding_folder
        self.transform = transform
        
        # List all image files (.jpg) and only keep ones with corresponding .npy files
        self.image_files = []
        self.embedding_files = []
        
        # Loop through all image files and check if the corresponding embedding exists
        for image_file in tqdm.tqdm(os.listdir(image_folder)):
            if len(self.image_files) > 10000:
                pass
            if image_file.endswith('.jpg'):
                embedding_file = image_file.replace('.jpg', '.npy')
                if os.path.exists(os.path.join(embedding_folder, embedding_file)):
                    self.image_files.append(image_file)
                    self.embedding_files.append(embedding_file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get the image and embedding filenames
        image_file = self.image_files[idx]
        embedding_file = self.embedding_files[idx]
        
        # Load the image
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        # Load the embedding (as a numpy array)
        embedding_path = os.path.join(self.embedding_folder, embedding_file)
        embedding = np.load(embedding_path)

        # Convert embedding to a tensor
        embedding = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

        return image, embedding

def get_last_saved_epoch(output_dir):
    """Get the last saved epoch number from the output directory."""
    if not os.path.exists(output_dir):
        return -1
    epoch_dirs = [d for d in os.listdir(output_dir) if re.match(r'epoch-\d+', d)]
    if not epoch_dirs:
        return -1
    last_epoch = max(int(re.search(r'epoch-(\d+)', d).group(1)) for d in epoch_dirs)
    return last_epoch

def load_model_from_last_epoch(output_dir, model, vqvae):
    """Load the model and VQ-VAE from the last saved epoch."""
    last_epoch = get_last_saved_epoch(output_dir)
    if last_epoch >= 0:
        epoch_dir = os.path.join(output_dir, f"epoch-{last_epoch}")
        model = UNet2DConditionModel.from_pretrained(epoch_dir)
        vqvae = VQModel.from_pretrained(epoch_dir)
    return model, vqvae, last_epoch

def compute_velocity(x_t, noise, timesteps, noise_scheduler):
    alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)  # Expand to match latents' dimensions
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    return alpha_t * noise - sqrt_one_minus_alpha_t * x_t

def train_loop(config, model, vqvae, noise_scheduler, optimizer, train_dataloader, lr_scheduler, eval_embedding, test_embedding):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        device_placement=True,
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    # model, vqvae, last_epoch = load_model_from_last_epoch(config.output_dir, model, vqvae)
    model, vqvae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, vqvae, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    # start_epoch = last_epoch + 1
    start_epoch = 0

    # Now you train the model
    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        progress_bar = tqdm.tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            images, embeddings = batch
            
            with torch.no_grad():
                latents = vqvae.encode(images).latents

            # Sample noise to add to the latents
            noise = torch.randn(latents.shape, device=latents.device)
            bs = latents.shape[0]

            # Sample a random timestep for each latent
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device,
                dtype=torch.int64
            )

            # Add noise to the latents (forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            # noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual

                # print(noisy_latents.shape)
                # print(embedding.shape)
                # Model predicts x0 directly
                # x0_pred = model(noisy_latents, timesteps, embeddings, return_dict=False)[0]
                # loss = F.mse_loss(x0_pred, latents)
                
                # Model predicts noise
                noise_pred = model(noisy_latents, timesteps, embeddings, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                
                # Model predict v_prediction
                # v_pred = model(noisy_latents, timesteps, embeddings, return_dict=False)[0]
                # v_true = compute_velocity(noisy_latents, noise, timesteps, noise_scheduler)
                # loss = F.mse_loss(v_pred, v_true)
                # accelerator.backward(loss)
                
                # print(f"Noise prediction mean: {noise_pred.mean()}, std: {noise_pred.std()}")
                # print(f"Target noise mean: {noise.mean()}, std: {noise.std()}")

                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            
        if epoch % 1 == 0:
            with torch.no_grad():
                holdout_latents = torch.randn(1, *latents.shape[1:], device=latents.device)  # Random initial latent

                # Reverse diffusion process to denoise
                for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
                    timesteps = torch.tensor([t], device=latents.device)
                    
                    # For noise predictions
                    noise_pred = model(holdout_latents, timesteps, eval_embedding, return_dict=False)[0]
                    holdout_latents = noise_scheduler.step(noise_pred, timesteps, holdout_latents).prev_sample
                    
                    # For latent predictions
                    # x0_pred = model(holdout_latents, timesteps, eval_embedding, return_dict=False)[0]
                    # # Manually compute x_t-1 from x0_pred and holdout_latents
                    # alpha_t = noise_scheduler.alphas_cumprod[t]
                    # alpha_t_prev = noise_scheduler.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=latents.device)

                    # beta_t = 1 - alpha_t
                    # beta_t_sqrt = beta_t.sqrt()

                    # # Formula for reverse step (x_{t-1})
                    # holdout_latents = (
                    #     alpha_t_prev.sqrt() * x0_pred +
                    #     beta_t_sqrt * (holdout_latents - alpha_t.sqrt() * x0_pred) / beta_t.sqrt()
                    # )
                    
                    # For v_prediction
                    # predicted_v = model(holdout_latents, timesteps, eval_embedding, return_dict=False)[0]

                    # # Compute noise from velocity
                    # alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                    # sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                    # predicted_noise = (predicted_v + sqrt_one_minus_alpha_t * holdout_latents) / alpha_t

                    # # Perform the reverse diffusion step
                    # holdout_latents = noise_scheduler.step(predicted_noise, timesteps, holdout_latents).prev_sample

                # Decode latents back to an image
                pred_image = vqvae.decode(holdout_latents).sample.squeeze(0).cpu().permute(1, 2, 0)
                pred_image = (pred_image * 127.5 + 127.5).clamp(0, 255).numpy().astype('uint8')  # Rescale to [0, 255]
                os.makedirs("eval_images_128", exist_ok=True)
                plt.imsave(f"eval_images_128/eval-{epoch}.jpg", pred_image)
                
                test_latents = torch.randn(1, *latents.shape[1:], device=latents.device)
                for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
                    timesteps = torch.tensor([t], device=test_embedding.device)
                    noise_pred = model(test_latents, timesteps, test_embedding, return_dict=False)[0]
                    test_latents = noise_scheduler.step(noise_pred, timesteps, test_latents).prev_sample
                test_image = vqvae.decode(test_latents).sample.squeeze(0).cpu().permute(1, 2, 0)
                test_image = (test_image * 127.5 + 127.5).clamp(0, 255).numpy().astype('uint8')
                plt.imsave(f"eval_images_128/eval-test-{epoch}.jpg", test_image)
        
        if accelerator.is_main_process:
            epoch_output_dir = os.path.join(config.output_dir, f"epoch-{epoch}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            model.save_pretrained(epoch_output_dir)

if __name__ == "__main__": 
    config = TrainingConfig()
    # Load pretrained models
    unet = UNet2DConditionModel(
        act_fn = "silu",
        attention_head_dim = 32,
        block_out_channels = [
            224,
            448,
            672,
            896
        ],
        center_input_sample = False,
        down_block_types = [
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D"
        ],
        downsample_padding = 1,
        flip_sin_to_cos = True,
        freq_shift = 0,
        in_channels = 3,
        layers_per_block = 2,
        mid_block_scale_factor = 1,
        norm_eps = 1e-05,
        norm_num_groups = 32,
        out_channels = 3,
        sample_size = 64,
        cross_attention_dim = 512,
        up_block_types = [
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D"
        ]
    )

    # unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet")
    vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    # scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler")
    scheduler = DDPMScheduler(
        beta_start=0.0015,
        beta_end=0.0195,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="epsilon",
        num_train_timesteps=1000
    )
    torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # unet = UNet2DConditionModel.from_pretrained("CompVis/ldm-text2im-large-256", subfolder="unet")
    # vqvae = VQModel.from_pretrained("CompVis/ldm-text2im-large-256", subfolder="vqvae")
    # scheduler = DDIMScheduler.from_config("CompVis/ldm-text2im-large-256", subfolder="scheduler")


    # Move to device
    # torch_device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # print(f"Device being used: {torch_device}")
    # unet.to(torch_device)
    # vqvae.to(torch_device)

    # Define dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),  # Resize images to 64x64 if needed
        transforms.ToTensor(),        # Convert PIL image to Tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    reverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5]),  # Revert normalization
        transforms.Lambda(lambda t: t.clamp(0, 1))  # Clamp to [0, 1]
    ])
    train_dataset = CelebADataset("data/img_align_celeba", "data/embeddings", transform=transform)
    # subset = Subset(train_dataset, range(config.train_batch_size*20))
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4)#, prefetch_factor=32)

    images, embeddings = next(iter(train_loader))
    eval_image = reverse_transform(images[0]).permute(1, 2, 0).numpy()
    eval_embedding = embeddings[0].unsqueeze(0).to(torch_device)

    test_embedding_path = "test_data/embeddings/001596.npy"
    test_embedding = torch.tensor(np.load(test_embedding_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(torch_device)

    os.makedirs("eval_images_128", exist_ok=True)
    plt.imsave("eval_images_128/base_image.jpg", eval_image)

    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_loader) * config.num_epochs),
    )
    
    train_loop(config, unet, vqvae, scheduler, optimizer, train_loader, lr_scheduler, eval_embedding, test_embedding)   
            
    # Save model
    unet.save_pretrained("models/128/small-unet")
    vqvae.save_pretrained("models/128/small-vqvae")
