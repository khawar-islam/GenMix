import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm
from tqdm import tqdm
import shutil
import csv

# -------------------------------
# Siamese DINOv2 Model
# -------------------------------
class SiameseDINOv2(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224.dino'):
        super(SiameseDINOv2, self).__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.embedding_dim = self.encoder.num_features

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, input1, input2):
        return self.forward_once(input1), self.forward_once(input2)

# -------------------------------
# Image Preprocessing
# -------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

# -------------------------------
# Filtering Logic
# -------------------------------
@torch.no_grad()
def classify_generated_by_similarity(
    original_root, generated_root,
    output_good, output_bad,
    log_path, min_similarity=0.7):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseDINOv2().to(device).eval()
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    os.makedirs(output_good, exist_ok=True)
    os.makedirs(output_bad, exist_ok=True)

    with open(log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class", "original_image", "generated_image", "similarity", "status"])

        for class_name in tqdm(os.listdir(original_root), desc="Processing classes"):
            orig_class_path = os.path.join(original_root, class_name)
            gen_class_path = os.path.join(generated_root, class_name)
            good_class_path = os.path.join(output_good, class_name)
            bad_class_path = os.path.join(output_bad, class_name)

            if not os.path.exists(gen_class_path):
                print(f"[{class_name}] No generated folder.")
                continue

            os.makedirs(good_class_path, exist_ok=True)
            os.makedirs(bad_class_path, exist_ok=True)

            for gen_img in os.listdir(gen_class_path):
                if "_generated_" not in gen_img:
                    continue

                try:
                    # recover original image name
                    orig_img = gen_img.split(".jpg_generated_")[0] + ".jpg"
                    orig_path = os.path.join(orig_class_path, orig_img)
                    gen_path = os.path.join(gen_class_path, gen_img)

                    if not os.path.exists(orig_path):
                        print(f"[{class_name}] Missing original: {orig_img}")
                        continue

                    img1 = load_image_tensor(orig_path).to(device)
                    img2 = load_image_tensor(gen_path).to(device)

                    emb1, emb2 = model(img1, img2)
                    emb1 = nn.functional.normalize(emb1, dim=1)
                    emb2 = nn.functional.normalize(emb2, dim=1)
                    sim = cos_sim(emb1, emb2).item()

                    status = "good" if sim >= min_similarity else "bad"
                    dest = os.path.join(good_class_path if status == "good" else bad_class_path, gen_img)
                    shutil.copy(gen_path, dest)

                    print(f"[{class_name}] {gen_img} → {status.upper()} (sim={sim:.4f})")
                    writer.writerow([class_name, orig_img, gen_img, round(sim, 4), status])

                except Exception as e:
                    print(f"[{class_name}] ERROR: {gen_img} → {e}")


if __name__ == "__main__":
    original_path = "/media/cvpr/CM_1/diffuseMix/result/original_resized/"
    generated_path = "/media/cvpr/CM_1/diffuseMix/result/generated/"
    output_good = "/media/cvpr/CM_1/diffuseMix/result/filtered_good/"
    output_bad = "/media/cvpr/CM_1/diffuseMix/result/filtered_bad/"
    csv_log_file = "/media/cvpr/CM_1/diffuseMix/result/similarity_log.csv"

    classify_generated_by_similarity(
        original_root=original_path,
        generated_root=generated_path,
        output_good=output_good,
        output_bad=output_bad,
        log_path=csv_log_file,
        min_similarity=0.7
    )
