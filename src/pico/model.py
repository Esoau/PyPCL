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
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args['moco_m'] + param_q.data * (1. - args['moco_m'])

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):
        """Update the queue of features and pseudo-labels."""
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
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1).detach()

        with torch.no_grad():
            self._momentum_update_key_encoder(args)
            _, k = self.encoder_k(im_k)

        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        
        self._dequeue_and_enqueue(k, pseudo_labels_b, args)
        return output, features, pseudo_labels, score_prot