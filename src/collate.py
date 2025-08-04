import torch

# Collate function for dataloading
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    
    # Get max length of labels
    max_len = max(len(label) for label in labels)
    
    # Pad label tensors to max length
    padded_labels = torch.full((len(labels), max_len), -1, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
        
    return images, padded_labels

def pico_collate_fn(batch):
    images_w, images_s, partial_Y, true_labels, indices = zip(*batch)
    images_w = torch.stack(images_w, 0)
    images_s = torch.stack(images_s, 0)
    partial_Y = torch.stack(partial_Y, 0)
    true_labels = torch.tensor(true_labels)
    indices = torch.tensor(indices)
    return images_w, images_s, partial_Y, true_labels, indices