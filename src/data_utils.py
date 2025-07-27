import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
    
class WeaklySupervisedDataset(Dataset):
    def __init__(self, data, targets, transform=None): # Add transform
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx]

class ComparisonDataGenerator:
    def __init__(self, ground_truth_dataset):
        self.dataset = ground_truth_dataset
        self.num_classes = len(self.dataset.classes)
        self.all_labels = np.arange(self.num_classes)

    def generate_pl_dataset(self, k: int):
        if not 1 < k <= self.num_classes:
            raise ValueError(f"'k' must be between 2 and {self.num_classes}.")
        new_targets = []
        original_data = self.dataset.data
        for _, true_label in tqdm(self.dataset, desc="Processing PL"):
            incorrect_labels = np.delete(self.all_labels, true_label)
            num_to_select = k - 1
            additional_candidates = np.random.choice(
                incorrect_labels, size=num_to_select, replace=False
            )
            candidate_set = np.append(additional_candidates, true_label)
            candidate_set.sort()
            new_targets.append(torch.tensor(candidate_set))
        return WeaklySupervisedDataset(original_data, new_targets)


    def generate_cl_dataset(self, m: int):
        if not 0 < m < self.num_classes:
            raise ValueError(f"'m' must be between 1 and {self.num_classes - 1}.")
        new_targets = []
        original_data = self.dataset.data
        for _, true_label in tqdm(self.dataset, desc="Processing CL"):
            incorrect_labels = np.delete(self.all_labels, true_label)
            complementary_set = np.random.choice(
                incorrect_labels, size=m, replace=False
            )
            complementary_set.sort()
            new_targets.append(torch.tensor(complementary_set))
        return WeaklySupervisedDataset(original_data, new_targets)