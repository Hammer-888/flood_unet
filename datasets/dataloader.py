import torch
import cv2
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import warnings

warnings.simplefilter("ignore", (UserWarning, FutureWarning))


def s1_to_rgb(vv_image, vh_image):
    ratio_image = np.clip(np.nan_to_num(vh_image / vv_image, 0), 0, 1)
    rgb_image = np.stack((vv_image, vh_image, 1 - ratio_image), axis=2)
    return rgb_image


class FloodDataset(Dataset):
    def __init__(self, train=True, crop_size=256):
        if train:
            self.root = "data/train"
        else:
            self.root = "data/val"
        self.vv_files = os.listdir(os.path.join(self.root, "vv"))[:1000]
        self.vh_files = os.listdir(os.path.join(self.root, "vh"))[:1000]
        self.label_files = os.listdir(os.path.join(self.root, "label"))[:1000]
        self.transform = Numpy2Torch()

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        fiel_path = os.path.join(self.root, "label", self.label_files[idx])
        image_vv = (
            cv2.imread(fiel_path.replace("label", "vv").replace(".png", "_vv.png"), 0)
            / 255.0
        )
        image_vh = (
            cv2.imread(fiel_path.replace("label", "vh").replace(".png", "_vh.png"), 0)
            / 255.0
        )
        rgb_image = s1_to_rgb(image_vv, image_vh).astype("float32")
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(fiel_path, 0) / 255.0
        image_vv, image_vh, label = self.transform(
            (
                image_vv.astype("float32"),
                image_vh.astype("float32"),
                label.astype("float32"),
            )
        )
        rgb_image = TF.to_tensor(rgb_image)
        simple = {"vv": image_vv, "vh": image_vh, "label": label, "rgb": rgb_image}
        return simple


def get_transforms(crop_size=256, train: bool = True):
    transformations = []
    if train:
        # transformations.append(Resize((crop_size * 1.2, crop_size * 1.2)))
        transformations.append(UniformCrop(crop_size))
        # transformations.append(RandomFlip)
        # transformations.append(RandomRotate)
        pass
    else:
        transformations.append(Resize((crop_size, crop_size)))
    transformations.append(Numpy2Torch)
    return transforms.Compose(transformations)


class Numpy2Torch(object):
    def __call__(self, sample: tuple):
        img1, img2, label = sample
        img1_tensor = TF.to_tensor(img1)
        img2_tensor = TF.to_tensor(img2)
        label_tensor = TF.to_tensor(label)
        return img1_tensor, img2_tensor, label_tensor


class RandomFlip(object):
    def __call__(self, sample):
        img1, img2, label = sample
        h_flip = np.random.choice([True, False])
        v_flip = np.random.choice([True, False])

        if h_flip:
            img1 = np.flip(img1, axis=1).copy()
            img2 = np.flip(img2, axis=1).copy()
            label = np.flip(label, axis=1).copy()

        if v_flip:
            img1 = np.flip(img1, axis=0).copy()
            img2 = np.flip(img2, axis=0).copy()
            label = np.flip(label, axis=0).copy()

        return img1, img2, label


class RandomRotate(object):
    def __call__(self, args):
        img1, img2, label = args
        k = np.random.randint(1, 4)  # number of 90 degree rotations
        img1 = np.rot90(img1, k, axes=(0, 1)).copy()
        img2 = np.rot90(img2, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        return img1, img2, label


# Performs uniform cropping on images
class UniformCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def random_crop(self, img1: np.ndarray, img2: np.ndarray, label: np.ndarray):
        height, width, _ = label.shape
        crop_limit_x = width - self.crop_size
        crop_limit_y = height - self.crop_size
        x = np.random.randint(0, crop_limit_x)
        y = np.random.randint(0, crop_limit_y)

        img1_crop = img1[
            y : y + self.crop_size,
            x : x + self.crop_size,
        ]
        img2_crop = img2[
            y : y + self.crop_size,
            x : x + self.crop_size,
        ]
        label_crop = label[
            y : y + self.crop_size,
            x : x + self.crop_size,
        ]
        return img1_crop, img2_crop, label_crop

    def __call__(self, sample: tuple):
        print(len(sample))
        img1, img2, label = sample
        img1, img2, label = self.random_crop(img1, img2, label)
        return img1, img2, label


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img1, img2, label = sample
        img1 = cv2.resize(img1, self.size, interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, self.size, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, self.size, interpolation=cv2.INTER_CUBIC)
        return img1, img2, label
