import torch
import torch.nn.functional as F
import torch.nn as nn

class partial_loss(nn.Module):
    def __init__(self, train_givenY):
        super().__init__()
        print('Calculating uniform targets...')
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        confidence = train_givenY.float()/tempY
        confidence = confidence.cuda()
        # calculate confidence
        self.confidence = confidence

    def forward(self, outputs, index, targets=None):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        if targets is None:
            # using confidence
            final_outputs = logsm_outputs * self.confidence[index, :].detach()
        else:
            # using given tagets
            final_outputs = logsm_outputs * targets.detach()
        loss_vec = - ((final_outputs).sum(dim=1))
        average_loss = loss_vec.mean()
        return average_loss, loss_vec

    @torch.no_grad()
    def confidence_update(self, temp_un_conf, batch_index):
        self.confidence[batch_index, :] = temp_un_conf
        return None