from albumentations import (
    HorizontalFlip, Normalize, Compose, Resize,
    ShiftScaleRotate, CoarseDropout
)
from albumentations.pytorch import ToTensorV2

def transformations():
    # Train Phase transformations
    train_transforms = Compose([
        Resize(32, 32),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value=None),
        HorizontalFlip(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2(),
    ])

    # Test Phase transformations
    test_transforms = Compose([
        Resize(32, 32),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2(),
    ])
    return train_transforms, test_transforms