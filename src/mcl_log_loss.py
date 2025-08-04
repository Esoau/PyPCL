import torch
import torch.nn as nn
import torch.nn.functional as F

class MCL_Log(nn.Module):
    def __init__(self, num_classes):
        super(MCL_Log, self).__init__()
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
