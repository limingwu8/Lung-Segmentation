import numpy as np
import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data.dataset import get_train_val_loader, inverse_normalize, get_test_loader
from model import UNet
from utils.Config import opt
from utils.vis_tool import Visualizer
from utils.eval_tool import compute_iou, save_pred_result
import utils.array_tool as at

def train(model, train_loader, criterion, epoch, vis):
    model.train()
    batch_loss = 0
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched['image']
        target = sample_batched['mask']
        data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        batch_loss += loss.data[0]
        if (batch_idx+1) % opt.plot_every == 0:
            ori_img_ = inverse_normalize(at.tonumpy(data[0]))
            target_ = at.tonumpy(target[0])
            pred_ = at.tonumpy(output[0])
            vis.img('gt_img', ori_img_)
            vis.img('gt_mask', target_)
            vis.img('pred_mask', (pred_ >= 0.5).astype(np.float32))

    batch_loss /= (batch_idx+1)
    print('epoch: ' + str(epoch) + ', train loss: ' + str(batch_loss))
    with open('logs.txt', 'a') as file:
        file.write('epoch: ' + str(epoch) + ', train loss: ' + str(batch_loss) + '\n')
    vis.plot('train loss', batch_loss)

def val(model, val_loader, criterion, epoch, vis):
    model.eval()
    batch_loss = 0
    avg_iou = 0
    for batch_idx, sample_batched in enumerate(val_loader):
        data = sample_batched['image']
        target = sample_batched['mask']
        data, target = Variable(data.type(opt.dtype), volatile=True), Variable(target.type(opt.dtype), volatile=True)
        output = model.forward(data)
        loss = criterion(output, target)
        batch_loss += loss.data[0]
        avg_iou += compute_iou(pred_masks=at.tonumpy(output >= 0.5).astype(np.float32), gt_masks=target)

    batch_loss /= (batch_idx+1)
    avg_iou /= len(val_loader.dataset)

    print('epoch: ' + str(epoch) + ', validation loss: ' + str(batch_loss), ', avg_iou: ', avg_iou)
    with open('logs.txt', 'a') as file:
        file.write('epoch: ' + str(epoch) + ', validation loss: ' + str(batch_loss) + ', avg_iou: ' + str(avg_iou) + '\n')

    vis.plot('val loss', batch_loss)
    vis.plot('validation average IOU', avg_iou)
    return avg_iou

# train and validation
def run(model, train_loader, val_loader, criterion, vis):
    best_iou = 0
    for epoch in range(1, opt.epochs+1):
        train(model, train_loader, criterion, epoch, vis)
        avg_iou = val(model, val_loader, criterion, epoch, vis)
        if avg_iou > best_iou:
            best_iou = avg_iou
            if opt.save_model:
                save_path = './checkpoints/RSNA_UNet_' + str(round(best_iou, 3)) + '_' + time.strftime('%m%d%H%M')
                torch.save(model.state_dict(), save_path)

    if opt.save_model:
        save_path = './checkpoints/RSNA_UNet_' + str(round(best_iou, 3)) + '_' + time.strftime('%m%d%H%M')
        torch.save(model.state_dict(), save_path)

# make prediction
def run_test(model, test_loader):
    pred_masks = []
    img_ids = []
    images = []
    for batch_idx, sample_batched in tqdm(enumerate(test_loader)):
        data, img_id = sample_batched['image'], sample_batched['img_id']
        data = Variable(data.type(opt.dtype), volatile=True)
        output = model.forward(data)
        # output = (output > 0.5)
        output = at.tonumpy(output)
        for i in range(0, output.shape[0]):
            pred_mask = np.squeeze(output[i])
            id = img_id[i]
            pred_mask = (pred_mask >= 0.5).astype(np.float32)
            pred_masks.append(pred_mask)
            img_ids.append(id)
            ori_img_ = inverse_normalize(at.tonumpy(data[i]))
            images.append(ori_img_)

    return img_ids, images, pred_masks

if __name__ == '__main__':
    """Train Unet model"""
    model = UNet(input_channels=1, nclasses=1)
    if opt.is_train:
        # split all data to train and validation, set split = True
        train_loader, val_loader = get_train_val_loader(opt.root_dir, batch_size=opt.batch_size, val_ratio=0.15,
                                                        shuffle=True, num_workers=4, pin_memory=False)

        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
        criterion = nn.BCELoss()
        vis = Visualizer(env=opt.env)

        if opt.is_cuda:
            model.cuda()
            criterion.cuda()
            if opt.n_gpu > 1:
                model = nn.DataParallel(model)

        run(model, train_loader, val_loader, criterion, vis)
    else:
        if opt.is_cuda:
            model.cuda()
            if opt.n_gpu > 1:
                model = nn.DataParallel(model)
        test_loader = get_test_loader(batch_size=20, shuffle=True,
                                      num_workers=opt.num_workers,
                                      pin_memory=opt.pin_memory)
        # load the model and run test
        model.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, 'RSNA_UNet_0.895_09210122')))

        img_ids, images, pred_masks = run_test(model, test_loader)

        save_pred_result(img_ids, images, pred_masks)