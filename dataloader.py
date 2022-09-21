import os

import torch
import torchio as tio
from glob2 import glob
from natsort import natsorted


def get_subjects(path):
    ct_files = natsorted(glob(os.path.join(path, 'data*')))
    label_files = natsorted(glob(os.path.join(path, 'label_*')))

    subjects = []
    for ct_file, label_file in zip(ct_files, label_files):
        subject = tio.Subject(
            ct=tio.ScalarImage(ct_file)
        )

        label = tio.LabelMap(label_file)
        subject.add_image(tio.LabelMap(tensor=label.data, affine=subject["ct"].affine), 'label_map')

        subjects.append(subject)

    return tio.SubjectsDataset(subjects)


def random_split(subjects, ratio=0.9):
    num_subjects = len(subjects)
    num_training_subjects = int(ratio * num_subjects)
    num_test_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_test_subjects
    return torch.utils.data.random_split(subjects, num_split_subjects)


class Dataset:
    def __init__(self, path, ratio=0.9, batch_size=1):
        self.path = path

        self.batch_size = batch_size

        self.training_transform = tio.Compose([
            tio.ToCanonical(),
            tio.RemapLabels({0: 0, 64: 1, 128: 2, 191: 3, 255: 4}),
            tio.RandomFlip(p=0.5),
            tio.Clamp(out_min=-500, out_max=500),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-500, 500)),
            tio.RandomAffine(scales=(0.9, 1.2), degrees=15, p=0.5),
            tio.OneHot(5),
        ])

        self.test_transform = tio.Compose([
            tio.ToCanonical(),
            tio.RemapLabels({0: 0, 64: 1, 128: 2, 191: 3, 255: 4}),
            tio.Clamp(out_min=-500, out_max=500),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-500, 500)),
            tio.OneHot(5),
        ])

        self.subjects = get_subjects(self.path)
        self.training_subjects, self.test_subjects = random_split(self.subjects, ratio)

        self.training_set = tio.SubjectsDataset(
            self.training_subjects, transform=self.training_transform)

        self.test_set = tio.SubjectsDataset(
            self.test_subjects, transform=self.test_transform)

    def __len__(self):
        return len(os.listdir(self.path) / 5)

    def get_loaders(self):
        training_loader = torch.utils.data.DataLoader(
            self.training_set, batch_size=self.batch_size,
            drop_last=True, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=self.batch_size,
            drop_last=True)

        print('Training set:', len(self.training_set), 'subjects')
        print('Test set:', len(self.test_set), 'subjects')

        return training_loader, test_loader
