import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pico.resnet import SupConResNet
from tqdm import tqdm

class PiCOModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder_q = SupConResNet(num_class=args['num_class'], feat_dim=args['low_dim'])
        self.encoder_k = SupConResNet(num_class=args['num_class'], feat_dim=args['low_dim'])

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(args['moco_queue'], args['low_dim']))
        self.register_buffer("queue_pseudo", torch.randn(args['moco_queue']))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("prototypes", torch.zeros(args['num_class'], args['low_dim']))
        self.queue = F.normalize(self.queue, dim=0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args['moco_m'] + param_q.data * (1. - args['moco_m'])

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert args['moco_queue'] % batch_size == 0
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % args['moco_queue']
        self.queue_ptr[0] = ptr

    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False):
        output, q = self.encoder_q(img_q)
        if eval_only:
            return output

        predicted_scores = torch.softmax(output, dim=1) * partial_Y
        max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)
        
        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(q, prototypes.t())
        score_prot = torch.softmax(logits_prot, dim=1)

        for feat, label in zip(q, pseudo_labels_b):
            self.prototypes[label] = self.prototypes[label] * args['proto_m'] + (1 - args['proto_m']) * feat
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder(args)
            _, k = self.encoder_k(im_k)

        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        
        self._dequeue_and_enqueue(k, pseudo_labels_b, args)
        return output, features, pseudo_labels, score_prot
    
def train_pico_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    model.train()
    total_loss = 0
    start_upd_prot = epoch >= pico_args['prot_start']
    
    progress_bar = tqdm(loader, desc=f"PiCO Epoch {epoch + 1}/{pico_args['epochs']}")
    for (images_w, images_s, partial_Y, true_labels, index) in progress_bar:
        images_w, images_s, partial_Y, index = images_w.to(device), images_s.to(device), partial_Y.to(device), index.to(device)
        
        cls_out, features, pseudo_target_cont, score_prot = model(images_w, images_s, partial_Y, pico_args)
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=partial_Y)
        
        mask = torch.eq(pseudo_target_cont[:batch_size].unsqueeze(1), pseudo_target_cont.unsqueeze(0)).float() if start_upd_prot else None

        loss_cls = loss_fn(cls_out, index)
        loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
        loss = loss_cls + pico_args['loss_weight'] * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    return total_loss / len(loader)