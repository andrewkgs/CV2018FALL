import numpy as np
import argparse
import cv2
import time
from scipy.ndimage import convolve
from scipy.ndimage.filters import gaussian_filter
from cv2 import ximgproc
from util import *
import imageio


parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('-il', '--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('-ir', '--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('-oe', '--output-edge', default='./data/Edge/TL0.png', type=str, help='output edge image')
parser.add_argument('-o', '--output', default='./TL0.pfm', type=str, help='left disparity map')
parser.add_argument('-d', '--max_disp', default=30, type=int, help='max dispairity value')
parser.add_argument('-gt', '--ground_truth', default=None, type=str, help='ground truth disparity map (available only for synthetic data)')
parser.add_argument('-ks', '--kernel_size', default=9, type=int)
args = parser.parse_args()


def preprocess(img, approach=0):
    if approach == 0:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    elif approach == 1:
        equ = cv2.equalizeHist(img)
        img_output = np.hstack(img, equ)
    elif approach == 2:
        img_output = cv2.Canny(img,100,200)
    elif approach == 3:
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        img_lab = cv2.merge(lab_planes)
        img_output = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    return img_output


def calculate_cost(I1, I2, Iedge, cost_type='best'):
    cost_d = np.zeros(shape=(I1.shape[0], I1.shape[1]))
    if cost_type == 'best':
        cost_d = np.clip(np.mean(np.abs(I1 - I2), axis=-1), a_min=None, a_max=15)
    elif cost_type == 'absolute':
        cost_d = np.mean(np.abs(I1 - I2), axis=-1)
    elif cost_type == 'squared':
        cost_d = np.mean(np.square(I1 - I2), axis=-1)
    elif cost_type == 'weighted':
        cost_d = np.clip(np.mean(np.abs(I1 - I2), axis=-1), a_min=None, a_max=15)
        cost_d = (Iedge/255 +2) * cost_d  
    else:
        print('Invalid cost type!')
        exit(0)
    return cost_d


def generate_kernel(size, kernel_type='best'):
    if kernel_type == 'best':
        kernel = np.full(shape=(size, size), fill_value=1.0/(size**2))
    elif kernel_type == 'uniform':
        kernel = np.full(shape=(size, size), fill_value=1.0/(size**2))
    elif kernel_type == 'gaussian':
        kernel_1D = cv2.getGaussianKernel(ksize=size, sigma=0.3)
        kernel = kernel_1D.T * kernel_1D
    return kernel


def computeDisp(Il, Ir, max_disp, Iedge):
    h, w, ch = Il.shape
    disp = np.zeros(shape=(h, w))
    Il, Ir, Iedge = Il.astype('float'), Ir.astype('float'), Iedge.astype('float') # the original dtype of Il and Ir is uint8
    # imageio.imwrite(args.output_edge, Iedge)
    # >>> Cost computation
    tic = time.time()
    # TODO: Compute matching cost from Il and Ir
    costs_original = np.zeros(shape=(h, w, max_disp))
    for d in range(1, max_disp+1):
        cost_d = calculate_cost(Il[:, d:], Ir[:, :-d], Iedge[:, d:], cost_type='weighted')
        costs_original[:, :, d-1] = np.pad(cost_d, pad_width=((0, 0), (d, 0)), mode='edge')

    toc = time.time()
    #print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    costs_aggregated = np.zeros(shape=(h, w, max_disp))
    kernel = generate_kernel(size=args.kernel_size, kernel_type='gaussian')
    for d in range(max_disp):
        costs_aggregated[:, :, d] = convolve(costs_original[:, :, d], kernel)
    toc = time.time()
    #print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    # >>> Disparity optimization
    tic = time.time()
    disp = np.argmin(costs_aggregated, axis=-1) + 1
    toc = time.time()
    #print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()

    # Cross-Shaped Voting
    # L, tau, = 17, 20
    # arm_span = np.zeros((h, w, 4), dtype=np.uint8) # hpd-, hpd+, vpd-, vpd+
    # print('Calculating arm_span of {0}...'.format('TL0'))
    # for y in range(h):
    #     for x in range(w):
    #         cont0, cont1, cont2, cont3 = True, True, True, True
    #         for ll in range(L):
    #             # print('arm_span:', d, y, x, ll)
    #             if cont0 and x  - ll >= 0:
    #                 Il_diff = max(np.abs(Il[y, x, 0] - Il[y, x - ll, 0]),
    #                                 np.abs(Il[y, x, 1] - Il[y, x - ll, 1]),
    #                                 np.abs(Il[y, x, 2] - Il[y, x - ll, 2]))
    #                 if Il_diff <= tau: arm_span[y, x, 0] = ll
    #                 else: cont0 = False
    #             else: cont0 = False
    #             if cont1 and x + ll < w:
    #                 Il_diff = max(np.abs(Il[y, x, 0] - Il[y, x + ll, 0]),
    #                                 np.abs(Il[y, x, 1] - Il[y, x + ll, 1]),
    #                                 np.abs(Il[y, x, 2] - Il[y, x + ll, 2]))
    #                 if Il_diff <= tau: arm_span[y, x, 1] = ll
    #                 else: cont1 = False
    #             else: cont1 = False
    #             if cont2 and y - ll >= 0:
    #                 Il_diff = max(np.abs(Il[y, x, 0] - Il[y - ll, x, 0]),
    #                                 np.abs(Il[y, x, 1] - Il[y - ll, x, 1]),
    #                                 np.abs(Il[y, x, 2] - Il[y - ll, x, 2]))
    #                 if Il_diff <= tau: arm_span[y, x, 2] = ll
    #                 else: cont2 = False
    #             else: cont2 = False
    #             if cont3 and y + ll < h:
    #                 Il_diff = max(np.abs(Il[y, x, 0] - Il[y + ll, x, 0]),
    #                                 np.abs(Il[y, x, 1] - Il[y + ll, x, 1]),
    #                                 np.abs(Il[y, x, 2] - Il[y + ll, x, 2]))
    #                 if Il_diff <= tau: arm_span[y, x, 3] = ll
    #                 else: cont3 = False
    #             else: cont3 = False
    #             if (cont0 or cont1 or cont2 or cont3) == False: break
    #         if x  > 0:    arm_span[y, x, 0] = max(arm_span[y, x, 0], 1)
    #         if x < w - 1: arm_span[y, x, 1] = max(arm_span[y, x, 1], 1)
    #         if y > 0:     arm_span[y, x, 2] = max(arm_span[y, x, 2], 1)
    #         if y < h - 1: arm_span[y, x, 3] = max(arm_span[y, x, 3], 1)

    # labels = np.zeros(shape=(h, w)).astype(np.int)
    # print('Calculating labels of {0}...'.format('TL0'))
    # for y in range(h):
    #     for x in range(w):
    #         histogram = np.array([], dtype=np.int64)
    #         for ver in range(y - arm_span[y, x, 2], y + arm_span[y, x, 3] + 1):
    #             for hor in range(x - arm_span[ver, x, 0], x + arm_span[ver, x, 1] + 1):
    #                 histogram = np.append(histogram, disp[ver, hor])
    #         counts = np.bincount(histogram)
    #         labels[y, x] = np.argmax(counts) 
    # disp = labels

    # Hole Filling
    k = 40
    for h_i in range(h):
        for w_i in range(w):
            if disp[h_i, w_i] == 1:
                neighbors = disp[max(0, h_i-k):min(h-1, h_i+k)+1, max(0, w_i-k):min(w-1, w_i+k)+1].flatten()
                disp[h_i, w_i] = np.argmax(np.bincount(neighbors))
    
    # Weighted Median Filtering
    disp = ximgproc.weightedMedianFilter(Il.astype('uint8'), disp.astype('float32'), 25, 5, ximgproc.WMF_JAC)

    toc = time.time()
    #print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    return disp


def main():
    print('Compute disparity for %s' % args.input_left)
    img_left = preprocess(cv2.imread(args.input_left), approach=0)
    img_right = preprocess(cv2.imread(args.input_right), approach=0)
    img_edge = preprocess(cv2.imread(args.input_left), approach=2)
    tic = time.time()
    disp = computeDisp(img_left, img_right, args.max_disp, img_edge)
    toc = time.time()
    writePFM(args.output, disp)
    #print('Elapsed time: %f sec.' % (toc - tic))
    if args.ground_truth != None:
        gt = readPFM(args.ground_truth)
        print('# average error: {}\n'.format(cal_avgerr(gt, disp)))


if __name__ == '__main__':
    main()
