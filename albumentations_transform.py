from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, CoarseDropout, 
    Normalize
)
from albumentations.pytorch import ToTensorV2

class AlbumentationsTransform:
    def __init__(self):
        self.train_transform = Compose([
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            CoarseDropout(
                max_holes=1, max_height=16, max_width=16,
                min_holes=1, min_height=16, min_width=16,
                fill_value=(0.4914, 0.4822, 0.4465),
                mask_fill_value=None,
                p=0.5
            ),
            Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
            ToTensorV2()
        ])

        self.test_transform = Compose([
            Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
            ToTensorV2()
        ])

    def __call__(self, img, train=True):
        if train:
            return self.train_transform(image=img)["image"]
        return self.test_transform(image=img)["image"]