import random
import numpy as np
from PIL import Image

def read_image(path, dtype=np.float32, color=True):
    """
    Read an image from a given path, the image is in (C, H, W) format and the range of its value is between [0, 255]
    :param path: The path of an image file
    :param dtype: data type of an image, default is float32
    :param color: If 'True', the number of channels is three, in this case, RGB
        If 'False', this function returns a gray scale image
    :return:
    """
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def flip_masks(masks, y_flip=False, x_flip=False):
    masks = masks.copy()
    for i in range(masks.shape[0]):
        if y_flip:
            masks[i] = np.flipud(masks[i]).copy()
        if x_flip:
            masks[i] = np.fliplr(masks[i]).copy()
    return masks

def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """
    Randomly flip an image in vertical or horizontal direction.
    :param img (numpy.ndarray): An array that gets flipped.
        This is in CHW format.
    :param y_random (bool): Randomly flip in vertical direction.
    :param x_random (bool): Randomly flip in horizontal direction.
    :param return_param (bool): Returns information of flip.
    :param copy (bool): If False, a view of :obj:`img` will be returned.
    :return (numpy.ndarray or (numpy.ndarray, dict)):
        If :`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.
        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.
        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.
    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


if __name__ == '__main__':
    pass