import os
import torch
import numpy as np
# from torchvision.utils import save_image
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import UNet2DConditionModel, VQModel, DDPMScheduler

# Paths
embeddings_dir = "./test_data/embeddings"
output_dir = "./test_data/predicted"
os.makedirs(output_dir, exist_ok=True)

# Load trained models
# vqvae_path = "models/vqvae"  # Replace with the actual path to your saved VQ-VAE
unet_path = "./ddpm-retrain/epoch-100"    # Replace with the actual path to your saved UNet
# vqvae = VQModel.from_pretrained(vqvae_path)
vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
unet = UNet2DConditionModel.from_pretrained(unet_path)
# scheduler = DDIMScheduler.from_config(unet_path)  # Load the diffusion scheduler config if needed
scheduler = DDPMScheduler(
        beta_start=0.0015,
        beta_end=0.0195,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="epsilon",
        num_train_timesteps=1000
    )
# Ensure models are on the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqvae.to(device)
unet.to(device)
scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

# Set the models to evaluation mode
vqvae.eval()
unet.eval()

# Define a function to decode latent codes to images
def decode_latents(latents):
    with torch.no_grad():
        images = vqvae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)  # Scale images to [0, 1]
    return images

# Update the inference function to handle batches
def inference_batch(embeddings):
    with torch.no_grad():
        batch_size = embeddings.size(0)
        latent_size = 16  # Based on VQ-VAE latent resolution

        # Initialize random latents for the batch
        holdout_latents = torch.randn(
            (batch_size, vqvae.config.latent_channels, latent_size, latent_size),
            device=device,
        )

        # Reverse diffusion process
        for t in reversed(range(scheduler.config.num_train_timesteps)):
            # timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)  # All timesteps in batch
            timesteps = torch.tensor([t], device=device)
            noise_pred = unet(holdout_latents, timesteps, embeddings, return_dict=False)[0]
            holdout_latents = scheduler.step(noise_pred, timesteps, holdout_latents).prev_sample

        # Decode latents to images
        pred_images = vqvae.decode(holdout_latents).sample  # [B, C, H, W]
        pred_images = pred_images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to [B, H, W, C]
        pred_images = (pred_images * 127.5 + 127.5).clip(0, 255).astype("uint8")  # Rescale to [0, 255]
        return pred_images

def predict_test_data():
    # Define the batch size
    batch_size = 10
    # Process embeddings in batches
    embedding_files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.npy')]*10)
    num_embeddings = len(embedding_files)

    # Loop through all embedding files in batches
    for batch_start in tqdm.tqdm(range(0, num_embeddings, batch_size), desc="Batch Processing"):
        batch_files = embedding_files[batch_start:batch_start + batch_size]
        embeddings = []

        # Load embeddings for the current batch
        for embedding_file in batch_files:
            embedding_path = os.path.join(embeddings_dir, embedding_file)
            embedding = np.load(embedding_path)
            embedding = torch.tensor(embedding, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Add dimensions
            embeddings.append(embedding)

        # Stack embeddings into a batch
        embeddings = torch.cat(embeddings, dim=0)  # [B, 1, 1, D]

        # Generate predictions for the batch
        pred_images = inference_batch(embeddings)  # [B, H, W, C]

        # Save all images in the batch
        for i, (embedding_file, pred_image) in enumerate(zip(batch_files, pred_images)):
            output_path = os.path.join(
                output_dir, f"{embedding_file.replace('.npy', '')}-{i:02d}.jpg"
            )
            plt.imsave(output_path, pred_image)

    print("All embeddings processed.")

from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import cv2
import warnings
warnings.filterwarnings("ignore")

def compare_predictions():
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(64, 64))
    embedding_files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.npy')])
    output_images = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    average_similarities = []
    training_similarities = []
    training_embeddings_dir = "./data/embeddings"
    training_embedding_files = sorted([f for f in os.listdir(training_embeddings_dir) if f.endswith('.npy')])
    all_training_embeddings = np.array([np.load(os.path.join(training_embeddings_dir, f)) 
                                        for f in training_embedding_files])
    print(f"Loaded {len(all_training_embeddings)} training embeddings.")
    
    for i, embedding_file in enumerate(embedding_files):
        embedding_path = os.path.join(embeddings_dir, embedding_file)
        original_embedding = np.load(embedding_path)
        
        cosine_sim_to_all = np.mean(cosine_similarity(
            np.expand_dims(original_embedding, axis=0),  # Query embedding
            all_training_embeddings                      # All preloaded embeddings
        ))
        print(f"Average cosine similarity to all for {embedding_file}: {cosine_sim_to_all}")
        training_similarities.append(cosine_sim_to_all)
        
        matching_images = [f for f in output_images if embedding_file.replace(".npy", "") in f]
        similarities = []
        for match in matching_images:
            image_path = os.path.join(output_dir, match)
            image = cv2.imread(image_path)
            faces = app.get(image)
            if len(faces) > 0:
                face = faces[0]
            else:
                print(f"Skipping {match}...")
                continue
            predicted_embedding = face.normed_embedding
            similarities.append(cosine_similarity(np.expand_dims(original_embedding, axis=0), np.expand_dims(predicted_embedding, axis=0))[0][0])
        print(f"{embedding_file}: {np.mean(similarities)}")
        average_similarities.append(np.mean(similarities))
    print(f"Total average: {np.mean(average_similarities)}")
    print(f"Distance to training: {np.mean(training_similarities)}")

if __name__ == "__main__":
    compare_predictions()