import torch
import torch.nn.functional as F
import torch.nn as nn

class partial_loss(nn.Module):
    def __init__(self, train_givenY, device):
        super().__init__()
        print('Calculating uniform targets...')
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        confidence = train_givenY.float()/tempY
        confidence = confidence.to(device)
        # Initial confidence is based on uniform distribution over candidate labels.
        self.confidence = confidence

    def forward(self, outputs, index, targets=None):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        if targets is None:
            # Use pre-computed confidence.
            final_outputs = logsm_outputs * self.confidence[index, :].detach()
        else:
            # Use given targets.
            final_outputs = logsm_outputs * targets.detach()
        loss_vec = - ((final_outputs).sum(dim=1))
        average_loss = loss_vec.mean()
        return average_loss, loss_vec

    @torch.no_grad()
    def confidence_update(self, temp_un_conf, batch_index):
        """Updates sample confidences."""
        self.confidence[batch_index, :] = temp_un_conf
        return None