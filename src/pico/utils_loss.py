import torch
import torch.nn.functional as F
import torch.nn as nn
    
class PartialLoss(nn.Module):
    def __init__(self, confidence):
        super().__init__()
        self.confidence = confidence
        self.conf_ema_m = 0.99

    def set_conf_ema_m(self, epoch, args):
        """Sets the EMA momentum for confidence updates based on the current epoch."""
        start = args['conf_ema_range'][0]
        end = args['conf_ema_range'][1]
        self.conf_ema_m = 1. * epoch / args['epochs'] * (end - start) + start

    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = -((final_outputs).sum(dim=1)).mean()
        return average_loss

    def confidence_update(self, temp_un_conf, batch_index, batchY):
        """Updates sample confidences using EMA."""
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().to(temp_un_conf.device).detach()
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] \
                                             + (1 - self.conf_ema_m) * pseudo_label

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss."""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = features.device
        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # Compute logits.
            anchor_dot_contrast = torch.div(torch.matmul(features[:batch_size], features.T), self.temperature)
            # For numerical stability.
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            # Mask-out self-contrast cases.
            logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
            mask = mask * logits_mask
            # Compute log_prob.
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
            # Compute mean of log-likelihood over positive samples.
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

            # Loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else: # MoCo Loss
            # Positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # Negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # Logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            # Apply temperature.
            logits /= self.temperature
            # Labels: positive key indicators.
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
            loss = F.cross_entropy(logits, labels)
        return loss