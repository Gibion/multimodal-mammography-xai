#import pydicom
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class CBISDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image (grayscale → RGB)
        img = cv2.imread(row["img_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {row['img_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Load mask (grayscale)
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {row['mask_path']}")

        # Resize both to same size BEFORE tensor conversion
        IMG_SIZE = 512
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype("float32")  # binary mask

        # Apply transforms to image only (they already include ToTensor)
        if self.transform is not None:
            img = self.transform(img)  # tensor [3, H, W]
        else:
            img = torch.from_numpy(img.astype("float32") / 255.).permute(2, 0, 1)

        # Convert mask to tensor [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Label
        label = torch.tensor(row["label"]).float()

        return img, mask, label
