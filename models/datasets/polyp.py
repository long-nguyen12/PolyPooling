import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

PALETTE = [[0, 0, 0], [255, 255, 255]]


class PolypDB(Dataset):
    def __init__(self, root: str, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.n_classes = 2
        self.ignore_label = -1

        img_path = Path(root) / "images"
        self.files = list(img_path.glob("*.png"))

        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} images.")

    @staticmethod
    def convert_to_mask(mask):
        h, w = mask.shape[:2]
        seg_mask = np.zeros((h, w, len(PALETTE)))
        for i, label in enumerate(PALETTE):
            seg_mask[:, :, i] = np.all(mask == label, axis=-1)
        return seg_mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace("images", "masks")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(lbl_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.convert_to_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)

            image = transformed["image"]
            mask = transformed["mask"]
            return image.float(), mask.argmax(dim=2).long()

        else:
            return image.float(), mask.argmax(dim=2).long()
