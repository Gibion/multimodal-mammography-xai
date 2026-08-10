#import pydicom
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# def preprocess_mammo(img, size=(224, 224)):
#     # Step 1 — Crop breast region using contour detection
#     img = crop_breast_contour(img)

#     # Step 2 — Remove pectoral muscle (optional)
#     img = remove_pectoral_muscle(img)

#     # Step 3 — Ensure grayscale
#     if img.ndim == 3:
#         img = img[:, :, 0]

#     # Step 4 — Convert to uint8 for CLAHE
#     img_uint8 = (img / img.max() * 255).astype("uint8")

#     # Step 5 — CLAHE
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     img_eq = clahe.apply(img_uint8)

#     # Step 6 — Resize
#     img_eq = cv2.resize(img_eq, size)

#     # Step 7 — Convert to RGB for ResNet
#     img_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)

#     # Step 8 — Normalize
#     img_rgb = img_rgb.astype("float32") / 255.0

#     return img_rgb


# def preprocess_mask(mask, size=(224, 224)):
#     """
#     Preprocess ROI mask:
#     - Ensure grayscale
#     - Resize
#     - Normalize to [0,1]
#     """
#     if mask.ndim == 3:
#         mask = mask[:, :, 0]

#     mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
#     mask = (mask > 0).astype("float32")  # binary mask

#     return mask

# class CBISDicomDataset(Dataset):
#     def __init__(self, metadata, transform=None, load_masks=True):
#         """
#         metadata must contain:
#         - actual_dicom_path
#         - actual_mask_path (optional)
#         - label
#         """
#         self.metadata = metadata
#         self.transform = transform
#         self.load_masks = load_masks

#     def __len__(self):
#         return len(self.metadata)

#     def __getitem__(self, idx):
#         row = self.metadata.iloc[idx]

#         # -------------------------
#         # Load DICOM image
#         # -------------------------
#         dcm_path = row["actual_dicom_path"]
#         ds = pydicom.dcmread(dcm_path)
#         img = ds.pixel_array.astype(np.float32)

#         # Preprocess image (CLAHE + resize + RGB)
#         img = preprocess_mammo(img)

#         # Convert to tensor (C,H,W)
#         img = torch.tensor(img).permute(2, 0, 1)

#         # -------------------------
#         # Load ROI mask (optional)
#         # -------------------------
#         if self.load_masks and "actual_mask_path" in row and pd.notna(row["actual_mask_path"]):
#             mask_path = row["actual_mask_path"]

#             try:
#                 if mask_path.endswith(".dcm"):
#                     ds_mask = pydicom.dcmread(mask_path)
#                     mask = ds_mask.pixel_array.astype(np.float32)
#                 else:
#                     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

#                 mask = preprocess_mask(mask)
#                 mask = torch.tensor(mask).unsqueeze(0)  # (1,H,W)

#             except:
#                 mask = torch.zeros((1, 224, 224), dtype=torch.float32)

#         else:
#             mask = torch.zeros((1, 224, 224), dtype=torch.float32)

#         # -------------------------
#         # Label
#         # -------------------------
#         label = torch.tensor(row["label"]).float()

#         return img, mask, label

# def remove_pectoral_muscle(img):
#     """
#     Removes the pectoral muscle from MLO mammograms using Hough line detection.
#     Works on grayscale images.
#     """
#     # Ensure grayscale
#     if img.ndim == 3:
#         img = img[:, :, 0]

#     # Normalize to uint8
#     img_uint8 = (img / img.max() * 255).astype("uint8")

#     # Edge detection
#     edges = cv2.Canny(img_uint8, 50, 150)

#     # Hough transform to detect strong diagonal line
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)

#     if lines is None:
#         return img  # no pectoral muscle detected

#     # Take the first detected line
#     rho, theta = lines[0][0]

#     # Convert line to points
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))

#     # Create mask for the triangular region
#     mask = np.zeros_like(img_uint8)

#     # Define triangle points (pectoral region)
#     pts = np.array([[0, 0], [x1, y1], [x2, y2]], np.int32)
#     pts = pts.reshape((-1, 1, 2))

#     cv2.fillPoly(mask, [pts], 255)

#     # Remove pectoral muscle by zeroing it out
#     img_removed = img.copy()
#     img_removed[mask == 255] = 0

#     return img_removed

# def crop_breast_region(img):
#     """
#     Automatically crop mammogram to the breast region by removing black background.
#     Works for CBIS-DDSM DICOMs.
#     """
#     # Create mask of non-zero pixels
#     mask = img > 0

#     # If the image is all zeros (rare), return original
#     if not mask.any():
#         return img

#     # Get bounding box of breast region
#     coords = np.argwhere(mask)
#     y0, x0 = coords.min(axis=0)
#     y1, x1 = coords.max(axis=0)

#     # Crop
#     cropped = img[y0:y1, x0:x1]

#     return cropped


# def crop_breast_contour(img):
#     """
#     Crop mammogram to breast region using contour detection.
#     Works for CBIS-DDSM DICOM pixel arrays.
#     """

#     # Normalize to uint8
#     img_uint8 = (img / img.max() * 255).astype(np.uint8)

#     # Threshold to separate breast from background
#     threshold = int(0.05 * img_uint8.max())
#     _, binary = cv2.threshold(img_uint8, threshold, 255, cv2.THRESH_BINARY)

#     # Find contours
#     contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) == 0:
#         return img  # fallback

#     # Select largest contour (breast)
#     areas = [cv2.contourArea(c) for c in contours]
#     breast_contour = contours[np.argmax(areas)]

#     # Convex hull for smooth boundary
#     breast_contour = cv2.convexHull(breast_contour)

#     # Bounding box
#     minx = breast_contour[:, 0, 0].min()
#     maxx = breast_contour[:, 0, 0].max()
#     miny = breast_contour[:, 0, 1].min()
#     maxy = breast_contour[:, 0, 1].max()

#     # Crop
#     cropped = img[miny:maxy, minx:maxx]

#     return cropped

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
