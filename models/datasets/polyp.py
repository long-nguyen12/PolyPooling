import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class PolypDB(Dataset):
    def __init__(self, root: str, transform=None) -> None:
        super().__init__()
        self.transform = transform

        img_path = Path(root) / "images"
        self.files = list(img_path.glob("*.png"))

        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} images.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace("images", "masks")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(lbl_path, 0)
        mask = mask[:, :, np.newaxis]
        mask = mask.astype("float32") / 255

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)

            image = transformed["image"]
            mask = transformed["mask"]
            return image.float(), mask.permute(2, 0, 1)

        else:
            return image.float(), mask.argmax(dim=2).long()
