import time
import os
import numpy as np
import matplotlib
import torch as t
import visdom
from skimage import io, transform

# matplotlib.use('Agg')
from matplotlib import pyplot as plot
from utils import array_tool as at
import matplotlib.patches as patches
from utils.Config import opt


VOC_BBOX_LABEL_NAMES = (
    'p'
)

def vis_image(img, img_id, ax=None):
    """Visualize a color image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """

    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(img_id)
    # CHW -> HWC
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax


def vis_bbox(img, img_id, bbox, label=None, score=None, ax=None):
    """Visualize bounding boxes inside image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(y_{min}, x_{min}, y_{max}, x_{max})` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """

    label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
    # add for index `-1`
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, img_id, ax=ax)
    # If there is no bounding box to display, visualize the image and exit.
    if bbox.size == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=1))

        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax

def vis_mask(img, bbox, mask, label, score=None, ax=None):
    """Visualize bounding boxes inside image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(y_{min}, x_{min}, y_{max}, x_{max})` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    img = img.copy()
    label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
    # add for index `-1`
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    color = [1, 0, 0]  # Red
    # Resize masks and patch them on the image
    for i in range(bbox.shape[0]):
        y1, x1, y2, x2 = int(bbox[i][0]), int(bbox[i][1]), int(bbox[i][2]), int(bbox[i][3])
        h = y2 - y1
        w = x2 - x1
        _mask = at.tonumpy(mask[i])
        if _mask.ndim==3:
            _mask = _mask[0]
        _mask = (transform.resize(_mask, (int(h), int(w)), preserve_range=False, mode='constant') > 0.5).astype(np.uint8)
        for c in range(3):
            img[c, y1:y2, x1:x2] = np.where(_mask==1,
                                            img[c, y1:y2, x1:x2]*(1 - 0.5) + 0.5*color[c] * 255,
                                            img[c, y1:y2, x1:x2])


    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=1))

        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax

def apply_mask_bbox(image, masks, bbox, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    ax = plot.subplot(111)
    ax.imshow(np.transpose(np.squeeze(image / 255.), (1, 2, 0)))
    for i in range(bbox.shape[0]):
        y1, x1, y2, x2 = int(bbox[i][0]), int(bbox[i][1]), int(bbox[i][2]), int(bbox[i][3])
        h = y2 - y1
        w = x2 - x1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        mask = at.tonumpy(masks[i])[0]
        mask = transform.resize(mask, (int(h), int(w)), preserve_range=False, mode='constant')
        for c in range(3):
            image[0, c, y1:y1+mask.shape[0], x1:x1+mask.shape[1]] = np.where(
                mask==1,
                image[0, c, y1:y1+mask.shape[0], x1:x1+mask.shape[1]]*(1 - 0.5) + alpha*color[c] * 255,
                image[0, c, y1:y1+mask.shape[0], x1:x1+mask.shape[1]])

    ax.imshow(np.transpose(np.squeeze(image / 255.), (1, 2, 0)))
    plot.show()

def fig2data(fig):
    """
    brief Convert a Matplotlib figure to a 4D numpy array with RGBA
    channels and return it

    @param fig： a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)


def fig4vis(fig):
    """
    convert figure to ndarray
    """
    ax = fig.get_figure()
    img_data = fig2data(ax).astype(np.int32)
    plot.close()
    # HWC->CHW
    return img_data[:, :, :3].transpose((2, 0, 1)) / 255.


def visdom_bbox(*args, **kwargs):
    fig = vis_bbox(*args, **kwargs)
    data = fig4vis(fig)
    return data


class Visualizer(object):
    """
    wrapper for visdom
    you can still access naive visdom function by
    self.line, self.scater,self._send,etc.
    due to the implementation of `__getattr__`
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self._vis_kw = kwargs

        # e.g.（’loss',23） the 23th value of loss
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        change the config of visdom
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        plot multi values
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            if v is not None:
                self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(t.Tensor(img_).cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def state_dict(self):
        return {
            'index': self.index,
            'vis_kw': self._vis_kw,
            'log_text': self.log_text,
            'env': self.vis.env
        }

    def load_state_dict(self, d):
        self.vis = visdom.Visdom(env=d.get('env', self.vis.env), **(self.d.get('vis_kw')))
        self.log_text = d.get('log_text', '')
        self.index = d.get('index', dict())
        return self

def rescale_back(img, gt_box, pred_box, scale):
    C, H, W = img.shape
    img = transform.resize(img, (C, H * (1 / scale), W * (1 / scale)), mode='reflect')
    o_H, o_W = H * (1 / scale), W * (1 / scale)
    gt_box = resize_bbox(gt_box, (H, W), (o_H, o_W))
    pred_box = resize_bbox(pred_box, (H, W), (o_H, o_W))
    return img, gt_box, pred_box


def save_gt_pred(img, gt_bbox, pred_bbox, pred_scores, img_id, save_path):
    fig = plot.figure()
    plot.title(img_id[0])
    ax1 = plot.subplot(121)
    ax1.imshow(np.transpose(img/255., (1, 2, 0)))
    ax1.set_title('gt: ' + img_id[0])
    # If there is no bounding box to display, visualize the image and exit.
    if len(gt_bbox) != 0:
        for i, bb in enumerate(gt_bbox):
            xy = (bb[1], bb[0])
            height = bb[2] - bb[0]
            width = bb[3] - bb[1]
            ax1.add_patch(plot.Rectangle(
                xy, width, height, fill=False, edgecolor='red', linewidth=1))


    ax2 = plot.subplot(122)
    ax2.imshow(np.transpose(img/255., (1, 2, 0)))
    ax2.set_title('pred: '+img_id[0])

    if len(pred_bbox) != 0:

        for i, bb in enumerate(pred_bbox):
            xy = (bb[1], bb[0])
            height = bb[2] - bb[0]
            width = bb[3] - bb[1]
            ax2.add_patch(plot.Rectangle(
                xy, width, height, fill=False, edgecolor='red', linewidth=1))

            caption = list()

            if pred_scores is not None:
                sc = pred_scores[i]
                caption.append('{:.2f}'.format(sc))

            if len(caption) > 0:
                ax2.text(bb[1], bb[0],
                        ': '.join(caption),
                        style='italic',
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})

    fig.set_size_inches(18.5, 10.5)
    plot.show()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plot.close()

def save_pred(img, pred_bbox, pred_scores, img_id, save_path):
    fig = plot.figure()
    ax1 = plot.subplot(111)
    ax1.imshow(np.transpose(img/255., (1, 2, 0)))
    ax1.set_title('pred: ' + img_id[0])

    if len(pred_bbox) != 0:

        for i, bb in enumerate(pred_bbox):
            xy = (bb[1], bb[0])
            height = bb[2] - bb[0]
            width = bb[3] - bb[1]
            ax1.add_patch(plot.Rectangle(
                xy, width, height, fill=False, edgecolor='red', linewidth=1))

            caption = list()

            if pred_scores is not None:
                sc = pred_scores[i]
                caption.append('{:.2f}'.format(sc))

            if len(caption) > 0:
                ax1.text(bb[1], bb[0],
                        ': '.join(caption),
                        style='italic',
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})

    fig.set_size_inches(18.5, 10.5)
    plot.show()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plot.close()