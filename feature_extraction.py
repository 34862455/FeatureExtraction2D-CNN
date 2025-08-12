import os
import json
import csv
import gzip
import pickle
import re
import shutil
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.io import read_image

# ------------------------ Config ------------------------
DATA_PATH = "/home/minneke/Documents/Dataset"
PHOENIX_PATH = "/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T"

EXTRACTOR = "efficientnet-b2" # Options "resnet50", "efficientnet-b0", "efficientnet-b2"
DATASET = "phoenix"  # Options: "phoenix", "sasl", "how2sign"
OUTPUT_NAME = f"{DATASET}_{EXTRACTOR}_imagenet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLITS = ["train", "dev", "test"]
BATCH_SIZE = 8

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ------------------------ Dataset Loaders ------------------------

def sort_key(x):
    match = re.search(r'(\d+)\.png$', x)
    if match:
        return int(match.group(1))
    print(f"⚠️ Skipping improperly named frame: {x}")
    return float('inf')

def make_dataset_phoenix(feature_root, annotation_file):
    dataset = []
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        next(csv_reader)
        for row in csv_reader:
            name, signer, gloss, text = row[0], row[4], row[5], row[6].lower()
            files = sorted(
                [f for f in os.listdir(os.path.join(feature_root, name)) if f.endswith('.png')],
                key=sort_key
            )
            dataset.append((files, name, signer, gloss, text))
    return dataset

def make_dataset_sasl(feature_root, annotation_file):
    dataset = []
    for row in annotation_file:
        name = row["file"].replace(".bag", "")
        if not os.path.exists(os.path.join(feature_root, name)):
            continue
        files = sorted([f for f in os.listdir(os.path.join(feature_root, name)) if f.endswith('.png')],
                       key=sort_key)
        text = row["trans"]
        dataset.append((files, name, "", "", text))
    return dataset

def make_dataset_how2sign(feature_root, annotation_file):
    dataset = []
    with open(annotation_file, encoding='cp850') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader)
        for row in csv_reader:
            name = row[3].strip()
            video_path = os.path.join(feature_root, name + ".mp4")
            if os.path.exists(video_path):
                text = row[6].lower()
                dataset.append(([], name, "", "", text))
    return dataset

def get_dataset_paths(dataset, split):
    if dataset == "phoenix":
        root = f"{DATA_PATH}/{PHOENIX_PATH}/features/fullFrame-210x260px/{split}"
        ann = f"{DATA_PATH}/{PHOENIX_PATH}/annotations/manual/PHOENIX-2014-T.{split}.corpus.csv"
    elif dataset == "sasl":
        root = f"{DATA_PATH}/SASL_Corpus_png_cropped/SASL Corpus png cropped"
        with open(f"{DATA_PATH}/SASL_Corpus_png_cropped/final_no_duplicates_text_num.json", "r") as f:
            ann_full = json.load(f)
        cut = int(len(ann_full) * 0.06)
        ann = {
            "train": ann_full[2*cut:],
            "dev": ann_full[0:cut],
            "test": ann_full[cut:2*cut],
        }[split]
    elif dataset == "how2sign":
        root = f"{DATA_PATH}/How2Sign/{split.capitalize()}/raw_videos"
        ann = f"{DATA_PATH}/How2Sign/{split.capitalize()}/how2sign_realigned_{split}.csv"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return root, ann

def load_dataset(dataset, root, ann):
    if dataset == "phoenix":
        return make_dataset_phoenix(root, ann)
    elif dataset == "sasl":
        return make_dataset_sasl(root, ann)
    elif dataset == "how2sign":
        return make_dataset_how2sign(root, ann)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

# ------------------------ Model Loader ------------------------
def load_resnet_model(device):
    from torchvision.models import ResNet50_Weights
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])  # remove FC layer → [B, 2048, 1, 1]
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_efficientnet_b0_model(device):
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])  # → [B, 1280, 1, 1]
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_efficientnet_b2_model(device):
    from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
    model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])  # → [B, 1408, 1, 1]
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

# ------------------------ Feature Extraction ------------------------
def extract_features_2dcnn(root, dataset, output_name, split, model, device, shard_size=1000):
    # Set up shard directory
    os.makedirs("data", exist_ok=True)
    shard_dir = f"data/tmp_{split}_shards"
    os.makedirs(shard_dir, exist_ok=True)

    resize = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    buffer = []
    shard_count = 0

    for idx, (files, name, signer, gloss, text) in enumerate(tqdm(dataset, desc=f"[{split}] Extracting with ResNet")):
        out = []
        for f in files:
            img_path = os.path.join(root, name, f)
            try:
                img = Image.open(img_path).convert('RGB')
                img = resize(img)
                img = to_tensor(img)
                img = normalize(img)
                img = img.unsqueeze(0).to(device)

                with torch.no_grad():
                    feat = model(img).squeeze().cpu()
                out.append(feat)
            except Exception as e:
                print(f"Failed: {img_path} — {e}")

        if len(out) < 2:
            continue

        out = torch.stack(out, dim=0)  # [T, 2048] for resnet [T, 1280] for efficientnet
        buffer.append({
            "name": name,
            "signer": signer,
            "gloss": gloss,
            "text": text,
            "sign": out.float()
        })

        # Save shard if buffer reaches limit
        if len(buffer) >= shard_size:
            shard_path = os.path.join(shard_dir, f"shard_{shard_count:04d}.pkl")
            with open(shard_path, "wb") as f:
                pickle.dump(buffer, f)
            buffer.clear()
            shard_count += 1
            torch.cuda.empty_cache()

    # Save final shard
    if buffer:
        shard_path = os.path.join(shard_dir, f"shard_{shard_count:04d}.pkl")
        with open(shard_path, "wb") as f:
            pickle.dump(buffer, f)

    # Merge shards to final file
    final_out = f"data/{output_name}_{split}.pt"
    merge_shards_and_save_final(shard_dir, final_out)


def merge_shards_and_save_final(shard_dir, final_output_path):
    """Merge sharded pickle files into one final .pt file."""
    all_data = []

    # Sort shards for correct order
    shard_files = sorted(
        [f for f in os.listdir(shard_dir) if f.startswith("shard_") and f.endswith(".pkl")]
    )

    for fname in tqdm(shard_files, desc=f"Merging {len(shard_files)} shards"):
        with open(os.path.join(shard_dir, fname), "rb") as f:
            shard_data = pickle.load(f)
            for item in shard_data:
                # Ensure tensor type and on CPU
                if not torch.is_tensor(item["sign"]):
                    item["sign"] = torch.tensor(item["sign"], dtype=torch.float32)
                else:
                    item["sign"] = item["sign"].cpu().float()
                all_data.append(item)

    # Save as a single compressed .pt file
    with gzip.open(final_output_path, "wb") as f:
        pickle.dump(all_data, f)

    print(f"Merged {len(all_data)} samples to {final_output_path}")

    # Remove temp shards
    shutil.rmtree(shard_dir)

# ------------------------ Main ------------------------
def main():
    if EXTRACTOR == "resnet50":
        model = load_resnet_model(device)
    elif EXTRACTOR == "efficientnet-b0":
        model = load_efficientnet_b0_model(device)
    elif EXTRACTOR == "efficientnet-b2":
        model = load_efficientnet_b2_model(device)
    else:
        raise ValueError(f"Unknown extractor: {EXTRACTOR}")

    for split in SPLITS:
        feature_root, annotation_file = get_dataset_paths(DATASET, split)
        dataset = load_dataset(DATASET, feature_root, annotation_file)

        extract_features_2dcnn(feature_root, dataset, OUTPUT_NAME, split, model, device)

if __name__ == "__main__":
    main()

