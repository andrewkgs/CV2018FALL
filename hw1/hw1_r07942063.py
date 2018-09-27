import os
import argparse
import numpy as np
from imageio import imread, imwrite
from cv2 import bilateralFilter


parser = argparse.ArgumentParser(description='CV hw1')
parser.add_argument('-d', '--data_dir', default='./testdata/', type=str)
args = parser.parse_args()


def load_image(path):
    if not os.path.exists(path):
        raise Exception("{} is not found.".format(path))
    return imread(path)


def con_rgb2gray(img_rgb):
    return np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


class JBF(object):
    def __init__(self, img, sigma_s, sigma_r):
        self.img = img # numpy array
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        self.weight = []
        self.guide_img = []
        self.generate_weight()
        self.generate_guide_img()

    def rd(self, num):
        return round(num, 1) + 0

    def generate_weight(self):
        for i in range(11):
            for j in range(11-i):
                self.weight.append([self.rd(i/10), self.rd(j/10), self.rd(1.0-(i/10)-(j/10))])

    def generate_guide_img(self):
        for w in self.weight:
            self.guide_img.append(np.dot(self.img[..., :3], w).astype(np.uint8))

    def jbf(self):
        bilateralFilter()
        for i in range(len(self.guide_img)):
            bilateralFilter(self.guide_img[i], )


def adv_rgb2gray(img):
    pass


def main():
    img_set = ['0a.png', '0b.png', '0c.png']
    sigma_s = [1, 2, 3]
    sigma_r = [0.05, 0.1, 0.2]
    for img_name in img_set:
        img = load_image(os.path.join(args.data_dir, img_name)) # in (R,G,B) order
        imwrite(img_name[:2] + '_y' + img_name[2:], con_rgb2gray(img))

        jbf = JBF(img, sigma_s, sigma_r)
        


if __name__ == '__main__':
    main()
