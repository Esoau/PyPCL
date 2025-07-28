import torch
import torch.nn as nn
import torch.nn.functional as F

class proden(nn.Module):
    def __init__(self):
        super(proden, self).__init__()
    def forward(self, outputs, partial_labels):
        # Create a mask to ignore padded labels (-1)
        mask = (partial_labels != -1)
        
        predictions = torch.softmax(outputs, dim=1)
        
        # Use the mask to select only valid labels for gathering
        masked_labels = partial_labels.clone()
        masked_labels[~mask] = 0 # Replace -1 with a valid index (0) to avoid gather errors
        
        candidate_preds = torch.gather(predictions, 1, masked_labels.long())
        candidate_preds[~mask] = 0 # Zero out the predictions for padded labels
        
        weights = candidate_preds / (torch.sum(candidate_preds, dim=1, keepdim=True) + 1e-8)
        
        log_probs = F.log_softmax(outputs, dim=1)
        individual_losses = -torch.gather(log_probs, 1, masked_labels.long())
        individual_losses[~mask] = 0 # Zero out the losses for padded labels
        
        sample_loss = torch.sum(weights * individual_losses, dim=1)
        return sample_loss.mean()

class LogURE(nn.Module):
    def __init__(self, num_classes):
        super(LogURE, self).__init__()
        self.num_classes = num_classes
    def forward(self, outputs, complementary_labels):
        # Create a mask to ignore padded labels (-1)
        valid_labels_mask = (complementary_labels != -1)

        batch_size, num_classes = outputs.shape
        probs_all = F.softmax(outputs, dim=1)
        
        mask_complementary = torch.zeros_like(probs_all, dtype=torch.bool)
        
        # Iterate over batch to handle variable-length labels
        for i in range(batch_size):
            valid_labels = complementary_labels[i][valid_labels_mask[i]]
            if len(valid_labels) > 0:
                mask_complementary[i].scatter_(0, valid_labels.long(), True)

        mask_non_complementary = ~mask_complementary
        sum_probs_not_in_complementary_set = (probs_all * mask_non_complementary.float()).sum(dim=1)
        epsilon = 1e-7
        sample_loss = -torch.log(sum_probs_not_in_complementary_set + epsilon)
        return sample_loss.mean()