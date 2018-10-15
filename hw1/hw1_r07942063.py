import os
import argparse
import numpy as np
from imageio import imread, imwrite


parser = argparse.ArgumentParser(description='CV hw1')
parser.add_argument('-d', '--data_dir', default='./testdata/', type=str)
parser.add_argument('-o', '--output_dir', default='./output/', type=str)
args = parser.parse_args()


def load_image(path):
    if not os.path.exists(path):
        raise Exception("{} is not found.".format(path))
    return imread(path)


def con_rgb2gray(img_rgb):
    return (np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114])).astype(np.uint8)


class JBF(object):
    def __init__(self, org_img, sigma_s, sigma_r):
        self.org_img = org_img
        self.img_h = self.org_img.shape[0]
        self.img_w = self.org_img.shape[1]
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        self.weight = []
        self.weight_10x = []
        self.guide_img = []
        self.Gr_LUT = {} # Look-Up Table
        self.r = 3*self.sigma_s
        self.window_size = 2*self.r + 1
        self.Gs_window = np.zeros((self.window_size, self.window_size))
        self.Gr_window = np.zeros((self.window_size, self.window_size))

        self.generate_weight()
        self.generate_guide_img()
        self.generate_Gr_LUT()
        self.Gs()


    def rd(self, num):
        return round(num, 1) + 0

    def generate_weight(self):
        for i in range(11):
            for j in range(11-i):
                self.weight.append([self.rd(i/10), self.rd(j/10), self.rd(1.0-(i/10)-(j/10))])
                self.weight_10x.append([i, j, 10-i-j])

    def generate_guide_img(self):
        for w in self.weight:
            self.guide_img.append(np.dot(self.org_img[..., :3], w))
        self.guide_img = np.array(self.guide_img)

    def generate_Gr_LUT(self):
        for i in range(0, 255):
            self.Gr_LUT[i] = np.exp(- (i**2) / (2 * (self.sigma_r**2)))         

    def Gs_weight(self, i, j):
        return np.exp(-1 * (np.sqrt((i-self.r)**2 + (j-self.r)**2)) / (2 * self.sigma_s**2))

    def Gs(self):
        for i in range(self.window_size):
            for j in range(self.window_size):
                self.Gs_window[i, j] = self.Gs_weight(i, j)

    def Gr_rgb(self, Tp, Tq):
        self.Gr_window = np.exp(-1 * ((Tp[0] - Tq[:,:,0])**2) / (2 * (self.sigma_r**2))) * \
                         np.exp(-1 * ((Tp[1] - Tq[:,:,1])**2) / (2 * (self.sigma_r**2))) * \
                         np.exp(-1 * ((Tp[2] - Tq[:,:,2])**2) / (2 * (self.sigma_r**2)))

    def Gr_gray(self, Tp, Tq):
        self.Gr_window = np.exp(-1 * ((Tp-Tq)**2) / (2 * (self.sigma_r**2)))

    def bf(self):
        r = self.r
        img_out = np.zeros((self.img_h - 2*r, self.img_w - 2*r, 3))
        for h in range(r, self.img_h-r):
            for w in range(r, self.img_w-r):
                self.Gr_rgb(self.org_img[h, w],
                            self.org_img[h-r:h+r+1, w-r:w+r+1])
                Iq = self.org_img[h-r:h+r+1, w-r:w+r+1]
                denominator = np.sum(np.multiply(self.Gs_window, self.Gr_window))
                img_out[h-r, w-r, 0] = np.sum(np.multiply(np.multiply(self.Gs_window, self.Gr_window), Iq[:,:,0])) / denominator
                img_out[h-r, w-r, 1] = np.sum(np.multiply(np.multiply(self.Gs_window, self.Gr_window), Iq[:,:,1])) / denominator
                img_out[h-r, w-r, 2] = np.sum(np.multiply(np.multiply(self.Gs_window, self.Gr_window), Iq[:,:,2])) / denominator
        return img_out


    def jbf(self, guide_id):
        r = self.r
        img_out = np.zeros((self.img_h - 2*r, self.img_w - 2*r, 3))
        for h in range(r, self.img_h-r):
            for w in range(r, self.img_w-r):
                self.Gr_gray(self.guide_img[guide_id, h, w],
                             self.guide_img[guide_id, h-r:h+r+1, w-r:w+r+1])
                Iq = self.org_img[h-r:h+r+1, w-r:w+r+1]
                denominator = np.sum(np.multiply(self.Gs_window, self.Gr_window))
                img_out[h-r, w-r, 0] = np.sum(np.multiply(np.multiply(self.Gs_window, self.Gr_window), Iq[:,:,0])) / denominator
                img_out[h-r, w-r, 1] = np.sum(np.multiply(np.multiply(self.Gs_window, self.Gr_window), Iq[:,:,1])) / denominator
                img_out[h-r, w-r, 2] = np.sum(np.multiply(np.multiply(self.Gs_window, self.Gr_window), Iq[:,:,2])) / denominator
        return img_out


    def run(self):
        if not os.path.exists('jbf'):
            os.makedirs('jbf')
        if not os.path.exists('bf'):
            os.makedirs('bf')
        bf_img = self.bf()
        imwrite('bf/0a.png', bf_img.astype(np.uint8))

        cost_dict = {}

        for guide_id in range(len(self.guide_img)):
            print('guide image no.{}'.format(guide_id))
            jbf_img = self.jbf(guide_id)
            imwrite('jbf/0a_{}.png'.format(guide_id), jbf_img.astype(np.uint8))
            cost = np.sum(np.abs(bf_img-jbf_img))
            print('cost: {}'.format(cost))
            print((self.weight[guide_id]))
            cost_dict[tuple(self.weight_10x[guide_id])] = cost

        counter = [0] * len(self.weight)
        for i, w in enumerate(self.weight_10x):
            for j, change in enumerate([[1, -1, 0], [1, 0, -1], [-1, 1, 0], [-1, 0, 1], [0, 1, -1], [0, -1, 1]]):
                neighbor = [sum(x) for x in zip(w, change)]
                if tuple(neighbor) in cost_dict:
                    if cost_dict[tuple(w)] >= cost_dict[tuple(neighbor)]:
                        break
                    else:
                        if j == 5:
                            counter[i] += 1
                        else:
                            continue
                else:
                    continue
        return counter


def main():
    img_set = ['0a.png', '0b.png', '0c.png']
    sigma_s_set = [1, 2, 3]
    sigma_r_set = [i*255 for i in [0.05, 0.1, 0.2]]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for img_name in img_set:
        img = load_image(os.path.join(args.data_dir, img_name)) # in (R,G,B) order
        imwrite(os.path.join(args.output_dir, img_name[:2] + '_y' + img_name[2:]), con_rgb2gray(img))

        total_vote = [0] * 66

        for sigma_s in sigma_s_set:
            for sigma_r in sigma_r_set:
                new_vote = JBF(img, sigma_s, sigma_r).run()
                total_vote = [sum(x) for x in zip(total_vote, new_vote)]
                print('img: {}\nsigma_s: {}, sigma_r: {}\n{}'.format(img_name, sigma_s, sigma_r, new_vote))
                print('current vote:\n{}')
        print(total_vote)

        with open('{}_vote_result.txt'.format(img_name[:2]), 'w') as fo:
            for v in total_vote:
                fo.write('{}\n'.format(v))


if __name__ == '__main__':
    main()
