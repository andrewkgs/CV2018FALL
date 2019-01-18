import numpy as np
import cv2
import time
from scipy.ndimage import convolve
from cv2 import ximgproc


def preprocess(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def calculate_cost(I1, I2, cost_type):
    cost_d = np.zeros(shape=(I1.shape[0], I1.shape[1]))
    if cost_type == 'best':
        cost_d = np.clip(np.mean(np.abs(I1 - I2), axis=-1), a_min=None, a_max=20)
    elif cost_type == 'Absolute':
        cost_d = np.mean(np.abs(I1 - I2), axis=-1)
    elif cost_type == 'Squared':
        cost_d = np.mean(np.square(I1 - I2), axis=-1)
    else:
        print('Invalid cost type!')
        exit(0)
    return cost_d


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros(shape=(h, w))
    Il, Ir = Il.astype('float'), Ir.astype('float') # the original dtype of Il and Ir is uint8

    # >>> Cost computation
    tic = time.time()
    # TODO: Compute matching cost from Il and Ir
    costs_original = np.zeros(shape=(h, w, max_disp))
    for d in range(1, max_disp+1):
        cost_d = calculate_cost(Il[:, d:], Ir[:, :-d], cost_type='best')
        costs_original[:, :, d-1] = np.pad(cost_d, pad_width=((0, 0), (d, 0)), mode='edge')

    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    costs_aggregated = np.zeros(shape=(h, w, max_disp))
    window = np.full(shape=(9, 9), fill_value=1.0/81.0)
    for d in range(max_disp):
        costs_aggregated[:, :, d] = convolve(costs_original[:, :, d], window)
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    # >>> Disparity optimization
    tic = time.time()
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    labels = np.argmin(costs_aggregated, axis=-1) + 1
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering

    # Hole Filling
    k=40
    for h_i in range(h):
        for w_i in range(w):
            if labels[h_i, w_i] == 1:
                neighbors = labels[max(0, h_i-k):min(h-1, h_i+k)+1, max(0, w_i-k):min(w-1, w_i+k)+1].flatten()
                labels[h_i, w_i] = np.argmax(np.bincount(neighbors))


    # Weighted Median Filtering
    labels = ximgproc.weightedMedianFilter(Il.astype('uint8'), labels.astype('float32'), 23, 5, ximgproc.WMF_JAC)
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    return labels.astype('uint8')


def main():
    print('Tsukuba')
    img_left = preprocess(cv2.imread('./testdata/tsukuba/im3.png'))
    img_right = preprocess(cv2.imread('./testdata/tsukuba/im4.png'))
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))

    print('Venus')
    img_left = preprocess(cv2.imread('./testdata/venus/im2.png'))
    img_right = preprocess(cv2.imread('./testdata/venus/im6.png'))
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('venus.png', np.uint8(labels * scale_factor))

    print('Teddy')
    img_left = preprocess(cv2.imread('./testdata/teddy/im2.png'))
    img_right = preprocess(cv2.imread('./testdata/teddy/im6.png'))
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))

    print('Cones')
    img_left = preprocess(cv2.imread('./testdata/cones/im2.png'))
    img_right = preprocess(cv2.imread('./testdata/cones/im6.png'))
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('cones.png', np.uint8(labels * scale_factor))


if __name__ == '__main__':
    main()
