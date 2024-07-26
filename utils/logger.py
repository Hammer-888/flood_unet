# adapted from https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard
from tensorboardX import SummaryWriter
from PIL import Image
import numpy as np


class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)

    def log_training(self, loss, precision, recall, oa, f1, iou, epoch):
        self.add_scalar("training/loss", loss, epoch)
        self.add_scalar("training/precision", precision, epoch)
        self.add_scalar("training/recall", recall, epoch)
        self.add_scalar("training/oa", oa, epoch)
        self.add_scalar("training/f1", f1, epoch)
        self.add_scalar("training/iou", iou, epoch)

    def log_validation(self, loss, precision, recall, oa, f1, iou, epoch):
        self.add_scalar("validation/loss", loss, epoch)
        self.add_scalar("validation/precision", precision, epoch)
        self.add_scalar("validation/recall", recall, epoch)
        self.add_scalar("validation/oa", oa, epoch)
        self.add_scalar("validation/f1", f1, epoch)
        self.add_scalar("validation/iou", iou, epoch)

    def log_images(self, rgb, vh, vv, target, prediction, epoch):
        # if len(rgb.shape) > 3:
        #     rgb = rgb.squeeze(0)
        # if len(target.shape) > 2:
        #     target = target.squeeze(0)
        # if len(prediction.shape) > 2:
        #     prediction = prediction.squeeze(0)
        # if len(vh.shape) > 3:
        #     vh = vh.squeeze(0)
        # if len(vv.shape) > 3:
        #     vv = vv.squeeze(0)
        self.add_image("vh", vh, epoch, dataformats="NCHW")
        self.add_image("vv", vv, epoch, dataformats="NCHW")
        self.add_image("rgb", rgb, epoch, dataformats="NCHW")
        self.add_image("mask", target, epoch, dataformats="NCHW")
        self.add_image("prediction", prediction, epoch, dataformats="NCHW")


class LogWriter(SummaryWriter):
    def __init__(self, logdir):
        super(LogWriter, self).__init__(logdir)

    def log_scaler(self, key, value, step, prefix="Training", helper_func=None):
        if helper_func:
            value = helper_func(value)
        self.add_scalar("{}/{}".format(prefix, key), value, step)

    def log_image(self, key, value, step, prefix="Training", helper_func=None):
        if helper_func:
            value = helper_func(value)
        self.add_image("{}/{}".format(prefix, key), value, step)
