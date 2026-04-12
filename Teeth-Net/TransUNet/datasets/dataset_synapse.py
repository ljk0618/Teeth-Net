import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_scale(image, label, scale_range=(0.8, 1.2)):
    h, w, _ = image.shape
    scale = random.uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)

    image_scaled = zoom(image, (scale, scale, 1), order=3)
    label_scaled = zoom(label, (scale, scale), order=0)

    if scale >= 1.0:
        sh = (new_h - h) // 2
        sw = (new_w - w) // 2
        image_out = image_scaled[sh:sh + h, sw:sw + w, :]
        label_out = label_scaled[sh:sh + h, sw:sw + w]
    else:
        pad_h = h - new_h
        pad_w = w - new_w
        ph1 = pad_h // 2
        ph2 = pad_h - ph1
        pw1 = pad_w // 2
        pw2 = pad_w - pw1
        image_out = np.pad(image_scaled,
                           ((ph1, ph2), (pw1, pw2), (0, 0)),
                           mode='constant', constant_values=0)
        label_out = np.pad(label_scaled,
                           ((ph1, ph2), (pw1, pw2)),
                           mode='constant', constant_values=0)
    return image_out, label_out


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']  # image: H×W×C, label: H×W

        r = random.random()
        if r < 0.33:
            image, label = random_rot_flip(image, label)
        elif r < 0.66:
            image, label = random_rotate(image, label)

        if random.random() < 0.5:
            image, label = random_scale(image, label, scale_range=(0.8, 1.2))

        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image,
                         (self.output_size[0] / x,
                          self.output_size[1] / y,
                          1),
                         order=3)
            label = zoom(label,
                         (self.output_size[0] / x,
                          self.output_size[1] / y),
                         order=0)

        if random.random() < 0.5:
            alpha = 1.0 + random.uniform(-0.1, 0.1)
            beta = random.uniform(-0.05, 0.05)
            image = image.astype(np.float32)
            image = image * alpha + beta
            image = np.clip(image, 0.0, 1.0)

        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2, 0, 1)  # HWC -> CHW
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        list_fp = os.path.join(list_dir, self.split + '.txt')
        with open(list_fp, 'r', encoding='utf-8-sig', errors='strict') as f:
            self.sample_list = [ln.strip() for ln in f if ln.strip()]
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir + "/" + slice_name + '.npz'
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir + "/" + slice_name + '.npz'
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image = torch.from_numpy(image.astype(np.float32))
            image = image.permute(2, 0, 1)
            label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
