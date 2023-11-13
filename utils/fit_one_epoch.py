import datetime
import torch
import os
from tqdm import tqdm
import numpy as np


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(epoch, epochs, optimizer, model, lr_scheduler, train_iter, val_iter, train_data_loader, val_data_loader, loss_history, save_period=2, save_dir='checkpoints'):
    loss_ep = 0
    train_data_loader.reset()
    print('---------------start training---------------')
    model.train()
    with tqdm(total=train_iter,desc=f'Epoch {epoch}/{epochs}',postfix=dict) as pbar:
        for i in range(train_iter):
            train_batch_loss = 0
            img, bboxes, labels = train_data_loader()
            targets = []
            for j in range(img.shape[0]):
                d = {}
                d['boxes'] = bboxes[j]
                d['labels'] = labels[j]
                targets.append(d)
            loss_dict = model(img, targets)
            train_batch_loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            train_batch_loss.backward()
            loss_ep += float(train_batch_loss.data.cpu().numpy())
            optimizer.step()
            pbar.set_postfix(**{'batch_loss'    : train_batch_loss, 
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)
            lr_scheduler.step()
    loss_ep = loss_ep / train_iter
    print('epoch_loss:', loss_ep)
    print('---------------start validate---------------')
    model.eval()
    with tqdm(total=val_iter,desc=f'Epoch {epoch}/{epochs}',postfix=dict) as pbar:
        val_batch_loss, bbox_loss, class_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i in range(val_iter):
                img, bboxes, labels = val_data_loader()
                pred = model(img)[0]
                pred_bbox = pred['boxes']
                pred_class = pred['labels']

                pbar.set_postfix(**{'val_loss'    : val_batch_loss/(i+1)})
                pbar.update(1)

    # print(f'\ntrain loss:{train_batch_loss} || val loss:{val_batch_loss/(val_iter)}\n')
    loss_history.append_loss(epoch + 1, loss_ep)
    # loss_history.append_loss(epoch + 1, train_batch_loss, val_loss / epoch_step_val)

    if epoch % save_period == 0 or epoch == epochs:
        torch.save(model.state_dict(), os.path.join('E:/ray_workspace/fasterrcnn_desnet/'+save_dir, f'ep{epoch}-loss{loss_ep}.pth'))
        # torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch, train_batch_loss)))
