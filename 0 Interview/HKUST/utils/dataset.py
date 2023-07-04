import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class SegmentaionDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, img_ext: str = 'jpg', mask_ext: str = 'png', mask_suffix: str = '_mask', transform: nn.Module = None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.mask_suffix = mask_suffix

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(f'.{self.img_ext}', f'{self.mask_suffix}.{self.mask_ext}'))

        # we used np.array, as opposed to PIL, as it will be converted to other formats (e.g., torch.Tensor) much easier
        # .convert('RGB') as img is RGB (3 channels)
        image = np.array(Image.open(img_path).convert('RGB'))
        # .convert('L') as mask is grayscale
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        # the mask's pixels are either 0.0 or 255. (black & white)
        # as we use sigmoid (for binary class) or softmax (for multiclass), then we need probability for each class (not 0.0 or 255.)
        mask = np.wehere(mask == 255., 1.0, mask)

        if self.transform:
            # TODO: add some augmentations (make sure it activates only during training, maybe should use nn.Module subclass)
            augmentations = self.transform(image=image, mask=mask)
            if isinstance(augmentations, (list, tuple)):
                image = augmentations[0]
                mask = augmentations[1]
            elif isinstance(augmentations, dict):
                image = augmentations['image']
                mask = augmentations['mask']

        return image, mask


###############################################################################
# For testing
###############################################################################
if __name__ == '__main__':
    pass