import torch
from tqdm import tqdm
from src.pico.model import PiCOModel
import torch.nn.functional as F
import numpy as np
import math
from src.solar.utils_algo import sinkhorn, linear_rampup

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, PiCOModel):
                outputs = model(images, eval_only=True)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_algorithm(model, loader, test_loader, loss_fn, optimizer, epochs, device):
    best_accuracy = 0.0
    accuracies = []
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
        avg_loss = total_loss / len(loader)
        current_accuracy = evaluate_model(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {current_accuracy:.2f}%")
        accuracies.append(current_accuracy)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
    print(f"Training finished. Best accuracy: {best_accuracy:.2f}%\n")
    return accuracies

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

def train_solar_epoch(solar_args, model, loader, loss_fn, optimizer, epoch, device, queue, emp_dist):
    model.train()
    total_loss = 0
    
    rho_start, rho_end = solar_args['rho_range']
    eta = solar_args['eta'] * linear_rampup(epoch, solar_args['warmup_epoch'])
    rho = rho_start + (rho_end - rho_start) * linear_rampup(epoch, solar_args['warmup_epoch'])

    progress_bar = tqdm(loader, desc=f"SoLar Epoch {epoch + 1}/{solar_args['epochs']}")
    for (images_w, images_s, partial_Y, true_labels, index) in progress_bar:
        images_w, images_s, partial_Y, index = images_w.to(device), images_s.to(device), partial_Y.to(device), index.to(device)

        logits_w = model(images_w)
        logits_s = model(images_s)
        bs = logits_w.shape[0]

        prediction = F.softmax(logits_w.detach(), dim=1)
        sinkhorn_cost = prediction * partial_Y
        
        # original code (memory leak)
        # prediction_queue = sinkhorn_cost.detach()

        # if queue is not None:
        #     if not torch.all(queue[-1, :] == 0):
        #         prediction_queue = torch.cat((queue, prediction_queue))
        #     # fill the queue
        #     queue[bs:] = queue[:-bs].clone().detach()
        #     queue[:bs] = prediction_queue[-bs:].clone().detach()

        # detach the sinkhorn_cost for queue operations.
        detached_sinkhorn_cost = sinkhorn_cost.detach()
        
        # create a temp variable for the sinkhorn algorithm input.
        sinkhorn_input = detached_sinkhorn_cost

        if queue is not None:
            if not torch.all(queue[-1, :] == 0):
                sinkhorn_input = torch.cat((queue, detached_sinkhorn_cost))

            queue[bs:] = queue[:-bs].clone().detach()
            queue[:bs] = detached_sinkhorn_cost.clone().detach()
        
        pseudo_label_soft, _ = sinkhorn(sinkhorn_input, solar_args['lamd'], r_in=emp_dist)
        
        pseudo_label = pseudo_label_soft[-bs:]
        pseudo_label_idx = pseudo_label.max(dim=1)[1]

        _, rn_loss_vec = loss_fn(logits_w, index)
        _, pseudo_loss_vec = loss_fn(logits_w, None, targets=pseudo_label)

        idx_chosen_sm = []
        sel_flags = torch.zeros(images_w.shape[0], device=device).detach()
        # initialize selection flags
        for j in range(solar_args['num_class']):
            indices = np.where(pseudo_label_idx.cpu().numpy()==j)[0]
            # torch.where will cause device error
            if len(indices) == 0:
                continue
                # if no sample is assigned this label (by argmax), skip
            bs_j = bs * emp_dist[j]
            pseudo_loss_vec_j = pseudo_loss_vec[indices]
            sorted_idx_j = pseudo_loss_vec_j.sort()[1].cpu().numpy()
            partition_j = max(min(int(math.ceil(bs_j*rho)), len(indices)), 1)
            # at least one example
            idx_chosen_sm.append(indices[sorted_idx_j[:partition_j]])

        if len(idx_chosen_sm) > 0:
            idx_chosen_sm = np.concatenate(idx_chosen_sm)
            sel_flags[idx_chosen_sm] = 1
        # filtering clean sinkhorn labels
        high_conf_cond = (pseudo_label * prediction).sum(dim=1) > solar_args['tau']
        sel_flags[high_conf_cond] = 1
        idx_chosen = torch.where(sel_flags == 1)[0]
        idx_unchosen = torch.where(sel_flags == 0)[0]

        if epoch < 1 or idx_chosen.shape[0] == 0:
            # first epoch, using uniform labels for training
            # else, if no samples are chosen, run rn 
            loss = rn_loss_vec.mean()
        else:
            if idx_unchosen.shape[0] > 0:
                loss_unreliable = rn_loss_vec[idx_unchosen].mean()
            else:
                loss_unreliable = 0
            loss_sin = pseudo_loss_vec[idx_chosen].mean()
            loss_cons, _ = loss_fn(logits_s[idx_chosen], None, targets=pseudo_label[idx_chosen])
            
            l = np.random.beta(4, 4)
            l = max(l, 1-l)
            X_w_c = images_w[idx_chosen]
            pseudo_label_c = pseudo_label[idx_chosen]
            rand_idx = torch.randperm(X_w_c.size(0))
            X_w_c_rand = X_w_c[rand_idx]
            pseudo_label_c_rand = pseudo_label_c[rand_idx]
            X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand        
            pseudo_label_c_mix = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
            logits_mix = model(X_w_c_mix)
            loss_mix, _  = loss_fn(logits_mix, None, targets=pseudo_label_c_mix)

            loss = (loss_sin + loss_mix + loss_cons) * eta + loss_unreliable * (1 - eta)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        conf_rn = sinkhorn_cost / sinkhorn_cost.sum(dim=1).repeat(prediction.size(1), 1).transpose(0, 1)
        loss_fn.confidence_update(conf_rn, index)
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    return total_loss / len(loader)