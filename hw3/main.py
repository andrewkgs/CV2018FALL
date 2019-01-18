import numpy as np
import cv2


def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    A = np.zeros((2*N, 8))
    for i in range(N):
        A[2*i, 0:2] = u[i]
        A[2*i, 2] = 1
        A[2*i, 6] = -(u[i, 0]*v[i, 0])
        A[2*i, 7] = -(u[i, 1]*v[i, 0])
        A[2*i+1, 3:5] = u[i]
        A[2*i+1, 5] = 1
        A[2*i+1, 6] = -(u[i, 0]*v[i, 1])
        A[2*i+1, 7] = -(u[i, 1]*v[i, 1])

    b = np.zeros((2*N, 1))
    for i in range(N):
        b[2*i] = v[i, 0]
        b[2*i+1] = v[i, 1]

    h = np.dot(np.linalg.inv(A), b)
    H = np.reshape(np.append(h, 1), (3, 3))

    return H


def transform(img, canvas, corners):
    h, w, ch = img.shape
    H = solve_homography(np.array(([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])), corners)
 
    for x in range(w):
        for y in range(h):
            XY_canvas = np.dot(H, np.array([[x], [y], [1]]))
            X_canvas = int(np.round(XY_canvas[0] / XY_canvas[2]))
            Y_canvas = int(np.round(XY_canvas[1] / XY_canvas[2]))
            canvas[Y_canvas, X_canvas] = img[y, x]
    return canvas


def backward_transform(img, canvas, corners):
    h, w, ch = canvas.shape
    H = solve_homography(np.array(([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])), corners)
 
    for x in range(w):
        for y in range(h):
            XY_img = np.dot(H, np.array([[x], [y], [1]]))
            X_img = int(np.round(XY_img[0] / XY_img[2]))
            Y_img = int(np.round(XY_img[1] / XY_img[2]))
            canvas[y, x] = img[Y_img, X_img]
    return canvas


def backward_transform_3D(img, canvas, corners, area):
    Y_max, X_max, _ = img.shape
    h, w, ch = canvas.shape
    H = solve_homography(area, corners)
 
    for x in range(w):
        for y in range(h):
            XY_img = np.dot(H, np.array([[x], [y], [1]]))
            X_img = int(np.round(XY_img[0] / XY_img[2]))
            Y_img = int(np.round(XY_img[1] / XY_img[2]))
            if (Y_max-1 >= Y_img >= 0) and (X_max-1 >= X_img >= 0):
                canvas[y, x] = img[Y_img, X_img]
    return canvas


def main():
    # Part 1
    canvas = cv2.imread('./input/times_square.jpg')
    img1 = cv2.imread('./input/Wang.jpg')
    img2 = cv2.imread('./input/Tu.jpg')
    img3 = cv2.imread('./input/Wu.jpg')
    img4 = cv2.imread('./input/Chien.jpg')
    img5 = cv2.imread('./input/Yang.jpg')

    corners1 = np.array([[818, 352], [884, 352], [818, 407], [885, 408]])
    corners2 = np.array([[311, 14], [402, 150], [157, 152], [278, 315]])
    corners3 = np.array([[364, 674], [430, 725], [279, 864], [369, 885]])
    corners4 = np.array([[808, 495], [892, 495], [802, 609], [896, 609]])
    corners5 = np.array([[1024, 608], [1118, 593], [1032, 664], [1134, 651]])

    # TODO
    canvas = transform(img1, canvas, corners1)
    canvas = transform(img2, canvas, corners2)
    canvas = transform(img3, canvas, corners3)
    canvas = transform(img4, canvas, corners4)
    canvas = transform(img5, canvas, corners5)
    cv2.imwrite('part1.png', canvas)

    # Part 2
    img = cv2.imread('./input/screen.jpg')
    # TODO
    corners_QR = np.array([[1039, 368], [1101, 394], [982, 553], [1035, 601]])
    QR_square = backward_transform(img, np.empty([300, 300, 3]), corners_QR)
    cv2.imwrite('part2.png', QR_square)

    # Part 3
    img_front = cv2.imread('./input/crosswalk_front.jpg')
    # TODO
    corners_crosswalk = np.array([[134, 162], [586, 157], [60, 240], [660, 229]])
    area_crosswalk = np.array([[85, 85], [464, 85], [85, 232], [464, 232]])
    crosswalk = np.empty([390, 550, 3])
    crosswalk = backward_transform_3D(img_front, crosswalk, corners_crosswalk, area_crosswalk)
    cv2.imwrite('part3.png', crosswalk)


if __name__ == '__main__':
    main()
