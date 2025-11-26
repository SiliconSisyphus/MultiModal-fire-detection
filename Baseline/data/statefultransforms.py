import torchvision.transforms.functional as functional500
import random

class StatefulRandomCrop(object):
    def __init__(self, insize, outsize):
        self.size = outsize
        self.cropParams = self.get_params(insize, self.size)

    @staticmethod
    def get_params(insize, outsize):

        w, h = insize
        th, tw = outsize
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):


        i, j, h, w = self.cropParams

        return functional.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class StatefulRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.rand = random.random()

    def __call__(self, img):

        if self.rand < self.p:
            return functional.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
