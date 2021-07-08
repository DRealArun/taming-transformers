import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        depth_paths = []
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        for path in paths:
            file_name = path.split('/')[-1].replace('RGB', 'Depth').replace('jpg', 'exr')
            depth_paths.append(os.path.join(path.split('/RGB')[0], file_name))
        self.data = ImagePaths(paths=paths, depth_paths=depth_paths, size=size, random_crop=False, depths=True)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        depth_paths = []
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        for path in paths:
            file_name = path.split('/')[-1].replace('RGB', 'Depth').replace('jpg', 'exr')
            depth_paths.append(os.path.join(path.split('/RGB')[0], file_name))
        self.data = ImagePaths(paths=paths, depth_paths=depth_paths, size=size, random_crop=False, depths=True)


