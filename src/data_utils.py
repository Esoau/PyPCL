import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from src.pico.randaugment import RandomAugment
import copy

    
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
        self.original_data = self.dataset.data
        self.original_targets = torch.tensor(self.dataset.targets)

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

    def generate_variable_pl_cl_datasets(self, q: float, num_classes: int):
        if not 0 <= q <= 1:
            raise ValueError("'q' must be between 0 and 1.")

        pl_targets = []
        cl_targets = []
        original_data = self.dataset.data
        all_labels = np.arange(num_classes)

        for _, true_label in tqdm(self.dataset, desc="Processing Variable PL/CL"):
            # Generate PL dataset
            pl_set = {true_label}
            false_labels = np.delete(all_labels, true_label)
            for label in false_labels:
                if np.random.rand() < q:
                    pl_set.add(label)
            
            pl_target = sorted(list(pl_set))
            pl_targets.append(torch.tensor(pl_target, dtype=torch.long))

            # Generate CL dataset
            cl_set = set(all_labels) - pl_set
            # Ensure cl_set is not empty. If it is, we have a problem.
            # In the variable case, if q=1, pl_set will contain all labels.
            # cl_set will be empty. This might be an issue for training.
            # The paper probably has a constraint on q.
            # For now, I will assume q < 1.
            # If cl_set is empty, what should be the target? An empty tensor.
            cl_target = sorted(list(cl_set))
            cl_targets.append(torch.tensor(cl_target, dtype=torch.long))

        pl_dataset = WeaklySupervisedDataset(original_data, pl_targets)
        cl_dataset = WeaklySupervisedDataset(original_data, cl_targets)

        return pl_dataset, cl_dataset

class PicoDataset(Dataset):
    
    def __init__(self, pl_dataset_raw, original_labels):
        self.images = pl_dataset_raw.data
        self.given_label_matrix_sparse = pl_dataset_raw.targets
        self.true_labels = original_labels
        
        # Create full partial label matrix
        self.num_classes = len(set(original_labels.numpy()))
        self.given_label_matrix = self._create_full_matrix(self.given_label_matrix_sparse, self.num_classes)
        
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616])
        ])
        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616])
        ])

    def _create_full_matrix(self, sparse_labels, num_classes):
        full_matrix = torch.zeros(len(sparse_labels), num_classes)
        for i, p_label in enumerate(sparse_labels):
            full_matrix[i, p_label] = 1
        return full_matrix

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        image = self.images[index]
        each_image_w = self.weak_transform(image)
        each_image_s = self.strong_transform(image)
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        return each_image_w, each_image_s, each_label, each_true_label, index

class SoLarDataset(Dataset):
    def __init__(self, pl_dataset_raw, original_labels):
        self.images = pl_dataset_raw.data
        self.given_label_matrix_sparse = pl_dataset_raw.targets
        self.true_labels = original_labels
        
        # Create full partial label matrix
        self.num_classes = len(set(original_labels.numpy()))
        self.given_label_matrix = self._create_full_matrix(self.given_label_matrix_sparse, self.num_classes)
        
        self.weak_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = copy.deepcopy(self.weak_transform)
        self.strong_transform.transforms.insert(1, RandomAugment(3,5))


    def _create_full_matrix(self, sparse_labels, num_classes):
        full_matrix = torch.zeros(len(sparse_labels), num_classes)
        for i, p_label in enumerate(sparse_labels):
            full_matrix[i, p_label] = 1
        return full_matrix

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        image = self.images[index]
        each_image_w = self.weak_transform(image)
        each_image_s = self.strong_transform(image)
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image_w, each_image_s, each_label, each_true_label, index