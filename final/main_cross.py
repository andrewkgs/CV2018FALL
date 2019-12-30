import numpy as np
import cv2
import time
import os.path
from scipy.signal import medfilt
from util import *

def computeDisp(Il, Ir, max_disp, name):
# def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.uint8)

    # >>> Cost computation    
    tic = time.time()
    # TODO: Compute matching cost from Il and Ir
    Il_f = Il.astype(float)
    Ir_f = Ir.astype(float)
    l = max_disp
    L, tau, T = 17, 20, 60

    filePath = './cross_data/{0}/SH.npy'.format(name)
    if not os.path.exists(filePath):
        SH = np.zeros((l, h, w))
        print('Calculating SH of {0}...'.format(name))
        for d in range(l):
            for y in range(h):
                for x in range(d, w):
                    # print('SH:', d, y, x)
                    abs_diff_sum = np.abs(Il_f[y, x, 0] - Ir_f[y, x - d, 0]) + np.abs(Il_f[y, x, 1] - Ir_f[y, x - d, 1]) + np.abs(Il_f[y, x, 2] - Ir_f[y, x - d, 2])
                    e_d = min(abs_diff_sum, T)
                    SH[d, y, x] = SH[d, y, max(0, x - 1)] + e_d
        np.save(filePath, SH)
    else:
        print('Loading SH of {0}...'.format(name))
        SH = np.load(filePath)

    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    filePath = './cross_data/{0}/arm_spans.npy'.format(name)
    if not os.path.exists(filePath):
        arm_spans = np.zeros((l, 4, h, w), dtype=np.uint8) # hpd-, hpd+, vpd-, vpd+
        print('Calculating arm_spans of {0}...'.format(name))
        for d in reversed(range(l)):
            for y in range(h):
                for x in range(d, w):
                    cont0, cont1, cont2, cont3 = True, True, True, True
                    for ll in range(L):
                        # print('arm_spans:', d, y, x, ll)
                        if cont0 and x - d - ll >= 0:
                            Il_diff = max(np.abs(Il_f[y, x, 0] - Il_f[y, x - ll, 0]),
                                            np.abs(Il_f[y, x, 1] - Il_f[y, x - ll, 1]),
                                            np.abs(Il_f[y, x, 2] - Il_f[y, x - ll, 2]))
                            Ir_diff = max(np.abs(Ir_f[y, x - d, 0] - Ir_f[y, x - d - ll, 0]),
                                            np.abs(Ir_f[y, x - d, 1] - Ir_f[y, x - d - ll, 1]),
                                            np.abs(Ir_f[y, x - d, 2] - Ir_f[y, x - d - ll, 2]))
                            if Il_diff <= tau and Ir_diff <= tau: arm_spans[d, 0, y, x] = ll
                            else: cont0 = False
                        else: cont0 = False
                        if cont1 and x + ll < w:
                            Il_diff = max(np.abs(Il_f[y, x, 0] - Il_f[y, x + ll, 0]),
                                            np.abs(Il_f[y, x, 1] - Il_f[y, x + ll, 1]),
                                            np.abs(Il_f[y, x, 2] - Il_f[y, x + ll, 2]))
                            Ir_diff = max(np.abs(Ir_f[y, x - d, 0] - Ir_f[y, x - d + ll, 0]),
                                            np.abs(Ir_f[y, x - d, 1] - Ir_f[y, x - d + ll, 1]),
                                            np.abs(Ir_f[y, x - d, 2] - Ir_f[y, x - d + ll, 2]))
                            if Il_diff <= tau and Ir_diff <= tau: arm_spans[d, 1, y, x] = ll
                            else: cont1 = False
                        else: cont1 = False
                        if cont2 and y - ll >= 0:
                            Il_diff = max(np.abs(Il_f[y, x, 0] - Il_f[y - ll, x, 0]),
                                            np.abs(Il_f[y, x, 1] - Il_f[y - ll, x, 1]),
                                            np.abs(Il_f[y, x, 2] - Il_f[y - ll, x, 2]))
                            Ir_diff = max(np.abs(Ir_f[y, x - d, 0] - Ir_f[y - ll, x - d, 0]),
                                            np.abs(Ir_f[y, x - d, 1] - Ir_f[y - ll, x - d, 1]),
                                            np.abs(Ir_f[y, x - d, 2] - Ir_f[y - ll, x - d, 2]))
                            if Il_diff <= tau and Ir_diff <= tau: arm_spans[d, 2, y, x] = ll
                            else: cont2 = False
                        else: cont2 = False
                        if cont3 and y + ll < h:
                            Il_diff = max(np.abs(Il_f[y, x, 0] - Il_f[y + ll, x, 0]),
                                            np.abs(Il_f[y, x, 1] - Il_f[y + ll, x, 1]),
                                            np.abs(Il_f[y, x, 2] - Il_f[y + ll, x, 2]))
                            Ir_diff = max(np.abs(Ir_f[y, x - d, 0] - Ir_f[y + ll, x - d, 0]),
                                            np.abs(Ir_f[y, x - d, 1] - Ir_f[y + ll, x - d, 1]),
                                            np.abs(Ir_f[y, x - d, 2] - Ir_f[y + ll, x - d, 2]))
                            if Il_diff <= tau and Ir_diff <= tau: arm_spans[d, 3, y, x] = ll
                            else: cont3 = False
                        else: cont3 = False
                        if (cont0 or cont1 or cont2 or cont3) == False: break
                    if x - d > 0: arm_spans[d, 0, y, x] = max(arm_spans[d, 0, y, x], 1)
                    if x < w - 1: arm_spans[d, 1, y, x] = max(arm_spans[d, 1, y, x], 1)
                    if y > 0:     arm_spans[d, 2, y, x] = max(arm_spans[d, 2, y, x], 1)
                    if y < h - 1: arm_spans[d, 3, y, x] = max(arm_spans[d, 3, y, x], 1)
        np.save(filePath, arm_spans)
    else:
        print('Loading arm_spans of {0}...'.format(name))
        arm_spans = np.load(filePath)
    
    filePath = './cross_data/{0}/EdH.npy'.format(name)
    if not os.path.exists(filePath):
        EdH = np.zeros((l, h, w))
        print('Calculating EdH of {0}...'.format(name))
        for d in range(l):
            for y in range(h):
                for x in range(w):
                    # print('EdH:', d, y, x)
                    if x - arm_spans[d, 0, y, x] <= 0:
                        EdH[d, y, x] = SH[d, y, x + arm_spans[d, 1, y, x]] # (x + arm_spans[d, 1, y, x]) should not be out of range
                    else:
                        EdH[d, y, x] = SH[d, y, x + arm_spans[d, 1, y, x]] - SH[d, y, x - arm_spans[d, 0, y, x] - 1]
        np.save(filePath, EdH)
    else:
        print('Loading EdH of {0}...'.format(name))
        EdH = np.load(filePath)
    
    filePath = './cross_data/{0}/SV.npy'.format(name)
    if not os.path.exists(filePath):
        SV = np.zeros((l, h, w))
        print('Calculating SV of {0}...'.format(name))
        for d in range(l):
            for y in range(h):
                for x in range(d, w):
                    # print('SV:', d, y, x)
                    EdH_n = EdH[d, y, x]
                    SV[d, y, x] = SV[d, max(0, y - 1), x] + EdH_n
        np.save(filePath, SV)
    else:
        print('Loading SV of {0}...'.format(name))
        SV = np.load(filePath)

    filePath = './cross_data/{0}/Ed.npy'.format(name)
    if not os.path.exists(filePath):
        Ed = np.zeros((l, h, w))
        print('Calculating Ed of {0}...'.format(name))
        for d in range(l):
            for y in range(h):
                for x in range(w):
                    # print('Ed:', d, y, x)
                    if y - arm_spans[d, 2, y, x] <= 0:
                        Ed[d, y, x] = SV[d, y + arm_spans[d, 3, y, x], x] # (y + arm_spans[d, 3, y, x]) should not be out of range
                    else:
                        Ed[d, y, x] = SV[d, y + arm_spans[d, 3, y, x], x] - SV[d, y - arm_spans[d, 2, y, x] - 1, x]
        np.save(filePath, Ed)
    else:
        print('Loading Ed of {0}...'.format(name))
        Ed = np.load(filePath)

    filePath = './cross_data/{0}/Ud.npy'.format(name)
    if not os.path.exists(filePath):
        Ud = np.zeros((l, h, w))
        print('Calculating Ud of {0}...'.format(name))
        for d in range(l):
            for y in range(h):
                for x in range(w):
                    for i in range(-int(arm_spans[d, 2, y, x]), int(arm_spans[d, 3, y, x] + 1)): # hpd-, hpd+, vpd-, vpd+
                        Ud[d, y, x] += (arm_spans[d, 0, y + i, x] + arm_spans[d, 1, y + i, x] + 1)
        np.save(filePath, Ud)
    else:
        print('Loading Ud of {0}...'.format(name))
        Ud = np.load(filePath)

    filePath = './cross_data/{0}/Ed_norm.npy'.format(name)
    if not os.path.exists(filePath):
        print('Calculating Ed_norm of {0}...'.format(name))
        Ed_norm = Ed / Ud
        np.save(filePath, Ed_norm)
    else:
        print('Loading Ed_norm of {0}...'.format(name))
        Ed_norm = np.load(filePath)
    
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    # >>> Disparity optimization
    tic = time.time()
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    filePath = './cross_data/{0}/raw_labels.npy'.format(name)
    if not os.path.exists(filePath):
        raw_labels = np.zeros((h, w), dtype=np.uint8)
        print('Calculating raw_labels of {0}...'.format(name))
        for y in range(h):
            for x in range(w):
                raw_labels[y, x] = np.argmin(Ed_norm[:, y, x][:(x + 1)])
        np.save(filePath, raw_labels)
    else:
        print('Loading raw_labels of {0}...'.format(name))
        raw_labels = np.load(filePath)

    filePath = './cross_data/{0}/arm_span.npy'.format(name)
    if not os.path.exists(filePath):
        arm_span = np.zeros((h, w, 4), dtype=np.uint8) # hpd-, hpd+, vpd-, vpd+
        print('Calculating arm_span of {0}...'.format(name))
        for y in range(h):
            for x in range(w):
                cont0, cont1, cont2, cont3 = True, True, True, True
                for ll in range(L):
                    # print('arm_span:', d, y, x, ll)
                    if cont0 and x  - ll >= 0:
                        Il_diff = max(np.abs(Il_f[y, x, 0] - Il_f[y, x - ll, 0]),
                                        np.abs(Il_f[y, x, 1] - Il_f[y, x - ll, 1]),
                                        np.abs(Il_f[y, x, 2] - Il_f[y, x - ll, 2]))
                        if Il_diff <= tau: arm_span[y, x, 0] = ll
                        else: cont0 = False
                    else: cont0 = False
                    if cont1 and x + ll < w:
                        Il_diff = max(np.abs(Il_f[y, x, 0] - Il_f[y, x + ll, 0]),
                                        np.abs(Il_f[y, x, 1] - Il_f[y, x + ll, 1]),
                                        np.abs(Il_f[y, x, 2] - Il_f[y, x + ll, 2]))
                        if Il_diff <= tau: arm_span[y, x, 1] = ll
                        else: cont1 = False
                    else: cont1 = False
                    if cont2 and y - ll >= 0:
                        Il_diff = max(np.abs(Il_f[y, x, 0] - Il_f[y - ll, x, 0]),
                                        np.abs(Il_f[y, x, 1] - Il_f[y - ll, x, 1]),
                                        np.abs(Il_f[y, x, 2] - Il_f[y - ll, x, 2]))
                        if Il_diff <= tau: arm_span[y, x, 2] = ll
                        else: cont2 = False
                    else: cont2 = False
                    if cont3 and y + ll < h:
                        Il_diff = max(np.abs(Il_f[y, x, 0] - Il_f[y + ll, x, 0]),
                                        np.abs(Il_f[y, x, 1] - Il_f[y + ll, x, 1]),
                                        np.abs(Il_f[y, x, 2] - Il_f[y + ll, x, 2]))
                        if Il_diff <= tau: arm_span[y, x, 3] = ll
                        else: cont3 = False
                    else: cont3 = False
                    if (cont0 or cont1 or cont2 or cont3) == False: break
                if x  > 0:    arm_span[y, x, 0] = max(arm_span[y, x, 0], 1)
                if x < w - 1: arm_span[y, x, 1] = max(arm_span[y, x, 1], 1)
                if y > 0:     arm_span[y, x, 2] = max(arm_span[y, x, 2], 1)
                if y < h - 1: arm_span[y, x, 3] = max(arm_span[y, x, 3], 1)
        np.save(filePath, arm_span)
    else:
        print('Loading arm_span of {0}...'.format(name))
        arm_span = np.load(filePath)

    filePath = './cross_data/{0}/labels.npy'.format(name)
    if not os.path.exists(filePath):
        # arm_span = np.zeros((h, w, 4), dtype=np.uint8) # hpd-, hpd+, vpd-, vpd+
        print('Calculating labels of {0}...'.format(name))
        for y in range(h):
            for x in range(w):
                histogram = np.array([], dtype=np.int64)
                for ver in range(y - arm_span[y, x, 2], y + arm_span[y, x, 3] + 1):
                    for hor in range(x - arm_span[ver, x, 0], x + arm_span[ver, x, 1] + 1):
                        histogram = np.append(histogram, raw_labels[ver, hor])
                counts = np.bincount(histogram)
                labels[y, x] = np.argmax(counts)                
        np.save(filePath, labels)
    else:
        print('Loading labels of {0}...'.format(name))
        labels = np.load(filePath)
        
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering

    # Median Filtering
    labels = np.array(medfilt(labels), dtype=np.int64)
    
    # Hole Filling
    k = 50
    for y in range(h):
        for x in range(w):
            if labels[y, x] == 0:
                neighbors = labels[max(0, y - k):min(h - 1, y + k) + 1, max(0, x - k):min(w - 1, x + k) + 1].reshape(-1)
                if np.sum(neighbors) == 0:
                    continue
                else:
                    counts = np.bincount(neighbors)
                    labels[y, x] = np.argmax(counts)               

    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    return labels


def main():
    img_left = cv2.imread('./cross_data/TL0_30/TL0.bmp')
    img_right = cv2.imread('./cross_data/TL0_30/TR0.bmp')
    max_disp = int(sys.argv[1])
    scale_factor = int(sys.argv[2])
    disp = computeDisp(img_left, img_right, max_disp, 'TL0_30')
    disp = disp.astype(np.float32)
    writePFM('TL0.pfm', disp)


if __name__ == '__main__':
    main()
