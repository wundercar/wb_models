import torch
from torch.utils.data import Dataset
from typing import List, Union
from PIL import Image


class ImagesDataset(Dataset):
    def __init__(self, images_paths: List[str], transforms=None):
        """
        Args:
            images_paths (List[string]): list of images paths
        """
        self.images_paths = images_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.images_paths)

    def open_img(self, path: str):
        image = Image.open(path)
        if self.transforms is not None:
            image = self.transforms(image)

        return image

    def __getitem__(self, idx: Union[int, List[int]]):
        images = []
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, int):
            return [self.open_img(self.images_paths[idx])]

        for index in idx:
            img = self.open_img(self.images_paths[index])
            images.append(img)

        return images
