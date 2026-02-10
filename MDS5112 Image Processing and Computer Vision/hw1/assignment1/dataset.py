import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class FacePreprocessConfig:
    enabled: bool = False
    margin: float = 0.45
    min_size: int = 20


class EG1800Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 224,
        augment: bool = False,
        face_preprocess: Optional[FacePreprocessConfig] = None,
        simulate_noncentered: bool = False,
        seed: int = 42,
    ):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "Images")
        self.labels_dir = os.path.join(root_dir, "Labels")
        self.split = split
        self.img_size = img_size
        self.augment = bool(augment) and split == "train"
        self.simulate_noncentered = bool(simulate_noncentered)

        self.rng = random.Random(seed)

        if face_preprocess is None:
            face_preprocess = FacePreprocessConfig(enabled=False)
        self.face_preprocess = face_preprocess

        self._face_detector = None
        if self.face_preprocess.enabled:
            if cv2 is None:
                raise ImportError(
                    "OpenCV (cv2) is required for face_preprocess, but it is not installed."
                )
            cascade_path = os.path.join(
                getattr(cv2.data, "haarcascades", ""),
                "haarcascade_frontalface_default.xml",
            )
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(
                    f"Cannot find OpenCV haarcascade file: {cascade_path}"
                )
            self._face_detector = cv2.CascadeClassifier(cascade_path)

        with open(os.path.join(root_dir, f"eg1800_{split}.txt"), "r") as f:
            self.image_names = [line.strip() for line in f]

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Mask transform
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
                transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long()),
            ]
        )

        self._color_jitter = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02
        )

    def __len__(self):
        return len(self.image_names)

    def _detect_face_bbox(
        self, image_rgb: Image.Image
    ) -> Optional[Tuple[int, int, int, int]]:
        if self._face_detector is None:
            return None
        np_img = np.array(image_rgb)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        faces = self._face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )
        if faces is None or len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        if min(w, h) < self.face_preprocess.min_size:
            return None
        return int(x), int(y), int(w), int(h)

    def _crop_around_face(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        bbox = self._detect_face_bbox(image.convert("RGB"))
        if bbox is None:
            return image, mask

        x, y, w, h = bbox
        img_w, img_h = image.size
        cx = x + w / 2.0
        cy = y + h / 2.0

        side = max(w, h) * (1.0 + self.face_preprocess.margin)
        left = int(round(cx - side / 2.0))
        top = int(round(cy - side / 2.0))
        right = int(round(cx + side / 2.0))
        bottom = int(round(cy + side / 2.0))

        left = max(0, left)
        top = max(0, top)
        right = min(img_w, right)
        bottom = min(img_h, bottom)

        if right - left < 2 or bottom - top < 2:
            return image, mask

        return image.crop((left, top, right, bottom)), mask.crop(
            (left, top, right, bottom)
        )

    def _maybe_simulate_noncentered(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if not self.simulate_noncentered:
            return image, mask

        # apply a random translation (same for image & mask)
        max_frac = 0.18
        dx = int(round(self.rng.uniform(-max_frac, max_frac) * image.size[0]))
        dy = int(round(self.rng.uniform(-max_frac, max_frac) * image.size[1]))
        image = TF.affine(
            image,
            angle=0.0,
            translate=(dx, dy),
            scale=1.0,
            shear=0.0,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.affine(
            mask,
            angle=0.0,
            translate=(dx, dy),
            scale=1.0,
            shear=0.0,
            interpolation=TF.InterpolationMode.NEAREST,
        )
        return image, mask

    def _maybe_augment(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if not self.augment:
            return image, mask

        # horizontal flip
        if self.rng.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # small rotation
        angle = self.rng.uniform(-10.0, 10.0)
        image = TF.rotate(
            image, angle=angle, interpolation=TF.InterpolationMode.BILINEAR
        )
        mask = TF.rotate(mask, angle=angle, interpolation=TF.InterpolationMode.NEAREST)

        # color jitter only for image
        image = self._color_jitter(image)

        return image, mask

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.labels_dir, img_name)).convert("L")

        if self.face_preprocess.enabled:
            image, mask = self._crop_around_face(image, mask)

        image, mask = self._maybe_simulate_noncentered(image, mask)
        image, mask = self._maybe_augment(image, mask)

        return self.img_transform(image), self.mask_transform(mask)
