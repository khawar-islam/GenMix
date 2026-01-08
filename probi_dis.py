import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import shutil

# === Step 0: Load DINOv2 Model ===
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dinov2 = dinov2.to(device)
print("✅ Using device:", device)

# === Step 1: Image Transformation ===
transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def get_sorted_image_files(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

def extract_features(folder, device):
    features = []
    filenames = get_sorted_image_files(folder)
    with torch.no_grad():
        for fname in tqdm(filenames, desc=f"Extracting from {folder}"):
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            feat = dinov2(img_tensor)
            features.append(feat.squeeze().cpu().numpy())
    return np.array(features), filenames

def gaussian_pdf(x, mean, std):
    coeff = 1 / (np.sqrt(2 * np.pi * std**2))
    exponent = np.exp(-((x - mean) ** 2) / (2 * std**2))
    return coeff * exponent

# === Step 2: Define Paths ===
original_dir = '/media/upendi/5602B27402B258A7/diffuseMix/result/original_resized/001.Black_footed_Albatross'
generated_dir = '/media/upendi/5602B27402B258A7/diffuseMix/result/generated/001.Black_footed_Albatross'
save_dir = 'result/dissimilar_generated_pdf/001.Black_footed_Albatross'
os.makedirs(save_dir, exist_ok=True)

# === Step 3: Extract Features ===
original_feats, original_filenames = extract_features(original_dir, device)

# === Step 4: Compute Intra-Original Similarities ===
intra_similarities = []
n = len(original_feats)
for i in range(n):
    for j in range(i + 1, n):
        sim = cosine_similarity([original_feats[i]], [original_feats[j]])[0][0]
        intra_similarities.append(sim)

intra_similarities = np.array(intra_similarities)
mean_sim = np.mean(intra_similarities)
std_sim = np.std(intra_similarities)

# PDF at mean (max density of Gaussian)
pdf_at_mean = gaussian_pdf(mean_sim, mean_sim, std_sim)
threshold = 0.01 * pdf_at_mean  # You can adjust this factor (0.01, 0.001, etc.)

print(f"\n📊 Mean similarity (original-original): {mean_sim:.4f}")
print(f"📈 STD: {std_sim:.4f}")
print(f"🔺 PDF at mean (max density): {pdf_at_mean:.6f}")
print(f"🔻 PDF-based dissimilarity threshold: {threshold:.6f}")

# === Step 5: Match Generated to Original and Filter ===
original_feat_dict = {fname: feat for fname, feat in zip(original_filenames, original_feats)}

generated_similarities = []
generated_pdfs = []
dissimilarities = []

for gen_fname in tqdm(get_sorted_image_files(generated_dir), desc="Matching Generated Images"):
    base_fname = gen_fname.split('.jpg')[0] + '.jpg'
    if base_fname not in original_feat_dict:
        print(f"⚠️ No matching original found for: {gen_fname}")
        continue

    try:
        img = Image.open(os.path.join(generated_dir, gen_fname)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            gen_feat = dinov2(img_tensor).squeeze().cpu().numpy()

        orig_feat = original_feat_dict[base_fname]
        sim = cosine_similarity([orig_feat], [gen_feat])[0][0]
        prob_density = gaussian_pdf(sim, mean_sim, std_sim)

        generated_similarities.append(sim)
        generated_pdfs.append(prob_density)

        # PDF-based filtering
        if prob_density < threshold:
            dissimilarities.append((gen_fname, sim, prob_density))
            shutil.copy(os.path.join(generated_dir, gen_fname), os.path.join(save_dir, gen_fname))

    except Exception as e:
        print(f"⚠️ Error processing {gen_fname}: {e}")

# === Step 6: Save Dissimilar Image Info ===
df = pd.DataFrame(dissimilarities, columns=["Filename", "Similarity", "PDF_Likelihood"])
df.to_csv("dissimilar_generated_with_pdf.csv", index=False)

# === Step 7: Plot Histograms ===
plt.figure(figsize=(10, 6))
plt.hist(intra_similarities, bins=30, alpha=0.6, label='Original-Original Similarities')
plt.hist(generated_similarities, bins=30, alpha=0.6, label='Original-Generated Similarities')
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Similarity Distribution (PDF-Based Filtering)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("similarity_pdf_distribution.png")
plt.show()

# === Step 8: Summary ===
print(f"\n🚨 Total dissimilar generated images saved (PDF < {threshold:.6f}): {len(dissimilarities)}")
print(f"📁 Saved to: {save_dir}")
