import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import transform
from utils.Config import opt
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import utils.array_tool as at
import matplotlib.patches as patches
from data.data_utils import read_image

DSB_BBOX_LABEL_NAMES = ('p')  # Pneumonia


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :].clip(min=0, max=255)
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

"""Transforms:
Data augmentation
"""
class Transform(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, in_data):
        img_id, img, mask = in_data['img_id'], in_data['image'], in_data['mask']
        _, H, W = img.shape
        img, mask = preprocess(img, mask, self.img_size)

        return {'img_id': img_id, 'image': img.copy(), 'mask': mask.copy()}


def preprocess(img, mask, img_size):
    C, H, W = img.shape
    img = img / 255.
    img = transform.resize(img, (C, img_size, img_size), mode='reflect')
    mask = mask.astype(np.float32)
    mask = transform.resize(mask, (1, img_size, img_size), mode='reflect')
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze

    img = normalize(img)

    return img, mask

def pytorch_normalze(img):
    """
    https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img

class RSNADataset(Dataset):
    def __init__(self, root_dir, img_id, mask_id, transform=True):
        """
        Args:
        :param root_dir (string): Directory with all the images
        :param img_id (list): lists of image id
        :param train: if equals true, then read training set, so the output is image, mask and imgId
                      if equals false, then read testing set, so the output is image and imgId
        :param transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.img_id = img_id
        self.mask_id = mask_id
        self.transform = transform
        self.tsf = Transform(opt.img_size)

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.img_id[idx].split('.')[0], 'image.png')
        mask_path = os.path.join(self.root_dir, 'masks', self.mask_id[idx])
        image = read_image(img_path, np.float32, False)
        mask = read_image(mask_path, np.uint8, False)

        sample = {'img_id': self.img_id[idx], 'image':image.copy(), 'mask':mask.copy()}

        if self.transform:
            sample = self.tsf(sample)

        return sample


class RSNADatasetTest(Dataset):
    def __init__(self, root_dir, transform=True):
        """
        Args:
        :param root_dir (string): Directory with all the images
        :param img_id (list): lists of image id
        :param train: if equals true, then read training set, so the output is image, mask and imgId
                      if equals false, then read testing set, so the output is image and imgId
        :param transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.img_id = os.listdir(root_dir)
        self.transform = transform
        self.tsf = Transform(opt.img_size)

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_id[idx], 'image.png')
        image = read_image(img_path, np.float32, False)

        C, H, W = image.shape
        image = image / 255.
        image = transform.resize(image, (C, opt.img_size, opt.img_size), mode='reflect')
        if opt.caffe_pretrain:
            normalize = caffe_normalize
        else:
            normalize = pytorch_normalze

        image = normalize(image)

        sample = {'img_id': self.img_id[idx], 'image': image.copy()}

        return sample

def get_train_loader(root_dir, batch_size=16, shuffle=False, num_workers=4, pin_memory=False):

    """Utility function for loading and returning training and validation Dataloader
    :param root_dir: the root directory of data set
    :param batch_size: batch size of training and validation set
    :param split: if split data set to training set and validation set
    :param shuffle: if shuffle the image in training and validation set
    :param num_workers: number of workers loading the data, when using CUDA, set to 1
    :param val_ratio: ratio of validation set size
    :param pin_memory: store data in CPU pin buffer rather than memory. when using CUDA, set to True
    :return:
        - train_loader: Dataloader for training
    """
    img_ids = os.listdir(root_dir)
    img_ids.sort()
    transformed_dataset = RSNADataset(root_dir=root_dir, img_id=img_ids, transform=True)
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader

def get_train_val_loader(root_dir, batch_size=16, val_ratio=0.2, shuffle=False, num_workers=4, pin_memory=False):

    """Utility function for loading and returning training and validation Dataloader
    :param root_dir: the root directory of data set
    :param batch_size: batch size of training and validation set
    :param split: if split data set to training set and validation set
    :param shuffle: if shuffle the image in training and validation set
    :param num_workers: number of workers loading the data, when using CUDA, set to 1
    :param val_ratio: ratio of validation set size
    :param pin_memory: store data in CPU pin buffer rather than memory. when using CUDA, set to True
    :return:
        - train_loader: Dataloader for training
        - valid_loader: Dataloader for validation
    """
    df = pd.read_csv(os.path.join(opt.root_dir, 'train.csv'))
    img_id, mask_id = list(df['image']), list(df['label'])
    train_img_id, val_img_id, train_mask_id, val_mask_id = train_test_split(img_id, mask_id, random_state=42, test_size=val_ratio, shuffle=False)

    train_dataset = RSNADataset(root_dir=root_dir, img_id=train_img_id, mask_id=train_mask_id, transform=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_dataset = RSNADataset(root_dir=root_dir, img_id=val_img_id, mask_id=val_mask_id, transform=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

def get_test_loader(batch_size=16, shuffle=False, num_workers=4, pin_memory=False):

    """Utility function for loading and returning training and validation Dataloader
    :param root_dir: the root directory of data set
    :param batch_size: batch size of training and validation set
    :param shuffle: if shuffle the image in training and validation set
    :param num_workers: number of workers loading the data, when using CUDA, set to 1
    :param pin_memory: store data in CPU pin buffer rather than memory. when using CUDA, set to True
    :return:
        - testloader: Dataloader of all the test set
    """
    transformed_dataset = RSNADatasetTest(root_dir=opt.test_root)
    testloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return testloader

def show_batch_train(sample_batched):
    """
    Visualize one training image and its corresponding bbox
    """
    img_id, image, mask = sample_batched['img_id'], sample_batched['image'], sample_batched['mask']
    image, mask = np.squeeze(at.tonumpy(image)), np.squeeze(at.tonumpy(mask))

    image = inverse_normalize(image)

    combined = np.multiply(image, mask)
    ax1 = plt.subplot(121)
    ax1.imshow(image / 255.)
    ax1.set_title(img_id[0])
    ax2 = plt.subplot(122)
    ax2.imshow(combined / 255.)
    ax2.set_title(img_id[0])
    plt.show()

def show_batch_test(sample_batch):
    img_id, image = sample_batch['img_id'], sample_batch['image']
    image = inverse_normalize(np.squeeze(at.tonumpy(image[0])))
    plt.figure()
    plt.imshow(image/255)
    plt.show()


if __name__ == '__main__':

    # Load training & validation set
    # train_loader, val_loader = get_train_val_loader(opt.root_dir, batch_size=1, val_ratio=0.2,
    #                                                 shuffle=False, num_workers=opt.num_workers,
    #                                                 pin_memory=opt.pin_memory)
    # for i_batch, sample in enumerate(val_loader):
    #     print(sample['img_id'], ', ', sample['image'].shape, ', ', sample['mask'].shape)
    #     show_batch_train(sample)

    test_loader = get_test_loader(batch_size=1, shuffle=False,
                                                num_workers=opt.num_workers,
                                                pin_memory=opt.pin_memory)
    for i_batch, sample in enumerate(test_loader):
        print(sample['img_id'], ', ', sample['image'].shape)
        show_batch_test(sample)
