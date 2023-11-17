import datetime
import torch
import os
import numpy as np
from tqdm import tqdm
from utils.score import loss_bce, miou


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(epoch, epochs, optimizer, model, lr_scheduler, train_iter, val_iter, train_data_loader, val_data_loader, loss_history, save_period=2, save_dir='checkpoints', device= 'cpu'):
    print('---------------start training---------------')
    loss_ep = 0
    model.train()
    with tqdm(total=train_iter,desc=f'Epoch {epoch}/{epochs}') as pbar:
        for img, label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            loss = loss_bce(output=output, target=label, mode='train')

            optimizer.zero_grad()
            loss.backward()
            loss_ep += float(loss.data.cpu().numpy())
            optimizer.step()

            pbar.set_postfix(**{'batch_loss'    : loss, 
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)
            lr_scheduler.step()
    loss_ep = loss_ep / train_iter
    print('---------------start validate---------------')
    val_loss = 0
    mious = 0
    model.eval()
    with tqdm(total=val_iter,desc=f'Epoch {epoch}/{epochs}',postfix=dict) as pbar:
        with torch.no_grad():
            for img, label in val_data_loader:
                img = img.to(device)
                label = label.to(device)

                output = model(img)

                loss = loss_bce(output=output, target=label, mode='val')
                val_loss += loss.data.cpu().numpy()
                
                mious += miou(output=output, target=label)

                pbar.update(1)

    val_loss    = val_loss / val_iter
    mious       = mious / val_iter

    print(f'\ntrain loss:{loss_ep} || val loss:{val_loss}, val_miou:{mious}\n')
    loss_history.append_loss(epoch + 1, loss_ep)
    loss_history.append_loss(epoch + 1, val_loss)

    if epoch % save_period == 0 or epoch == epochs:
        torch.save(model.state_dict(), os.path.join('E:/ray_workspace/fasterrcnn_desnet/'+save_dir, f'ep{epoch}-loss{loss_ep}.pth'))
