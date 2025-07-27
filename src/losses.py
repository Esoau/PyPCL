import torch
import torch.nn as nn
import torch.nn.functional as F

class proden(nn.Module):
    def __init__(self):
        super(proden, self).__init__()
    def forward(self, outputs, partial_labels):
        predictions = torch.softmax(outputs, dim=1)
        candidate_preds = torch.gather(predictions, 1, partial_labels.long())
        weights = candidate_preds / (torch.sum(candidate_preds, dim=1, keepdim=True) + 1e-8)
        log_probs = F.log_softmax(outputs, dim=1)
        individual_losses = -torch.gather(log_probs, 1, partial_labels.long())
        sample_loss = torch.sum(weights * individual_losses, dim=1)
        return sample_loss.mean()

class LogURE(nn.Module):
    def __init__(self, num_classes):
        super(LogURE, self).__init__()
        self.num_classes = num_classes
    def forward(self, outputs, complementary_labels):
        batch_size, num_classes = outputs.shape
        probs_all = F.softmax(outputs, dim=1)
        mask_complementary = torch.zeros_like(probs_all, dtype=torch.bool).scatter_(1, complementary_labels.long(), True)
        mask_non_complementary = ~mask_complementary
        sum_probs_not_in_complementary_set = (probs_all * mask_non_complementary.float()).sum(dim=1)
        epsilon = 1e-7
        sample_loss = -torch.log(sum_probs_not_in_complementary_set + epsilon)
        return sample_loss.mean()