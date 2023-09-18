import numpy as np
import cv2 as cv


def crop_img_f(img):
    center = img.shape
    h = int(img.shape[0] * 0.75)
    w = int(img.shape[1] * 0.75)

    x = center[1] / 2 - w / 2
    y = center[0] / 2 - h / 2

    crop_img = img[int(y):int(y + h), int(x):int(x + w)]

    return crop_img


def apply_threshold(img):
    th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 5)
    return th


def dilate_image(img):
    kernel = np.ones((5, 5), np.uint8)

    dilation = cv.dilate(img, kernel, iterations=1)

    return dilation


def get_derievetives(img):
    ddepth = cv.CV_8UC1
    return cv.Sobel(img, ddepth, 0, 1), cv.Sobel(img, ddepth, 1, 0)


def get_horizontal_lines(dv):
    lines = cv.HoughLines(dv, 1, np.pi / 180, 130, None, min_theta=1.2, max_theta=1.8)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(lines, 2, None, criteria, 35, cv.KMEANS_RANDOM_CENTERS)

    A = lines[label.ravel() == 0]
    B = lines[label.ravel() == 1]

    return np.vstack(([np.median(A, axis=0)], [np.median(B, axis=0)]))


def get_vertical_lines(dv):
    lines_angle_left = cv.HoughLines(dv, 1, np.pi / 180, 130, None, 0, max_theta=0.2)
    lines_angle_right = cv.HoughLines(dv, 1, np.pi / 180, 130, None, 0, min_theta=3)

    if lines_angle_right is not None and lines_angle_left is not None:
        lines = np.concatenate((lines_angle_right, lines_angle_left))
    else:
        lines = lines_angle_right if lines_angle_right is not None else lines_angle_left

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(np.abs(lines[:, :, 0]), 2, None, criteria, 35, cv.KMEANS_RANDOM_CENTERS)

    A = lines[label.ravel() == 0]
    B = lines[label.ravel() == 1]

    return np.vstack(([np.median(A, axis=0)], [np.median(B, axis=0)]))


def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def get_intersections(horiz_lines, vert_lines):
    points = []
    for h_l in sorted(horiz_lines, key=lambda line: line[0][0]):
        for v_l in sorted(vert_lines, key=lambda line: line[0][0]):
            points.append(intersection(h_l, v_l))

    return np.array(points)


def get_sections(image, points):
    intersection_points = sorted(points, key=lambda p: p[0])

    for point in intersection_points:
        point[0] += int(image.shape[1] * 0.25 / 2)
        point[1] += int(image.shape[0] * 0.25 / 2)

    p1, p3 = sorted(intersection_points[:2], key=lambda p: p[1])
    p2, p4 = sorted(intersection_points[2:], key=lambda p: p[1])

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    return [
        image[:y1, :x1],
        image[:y1, x1:x2],
        image[:y1, x2:],

        image[y1:y3, :x3],
        image[y1:y3, x3:x4],
        image[y2:y4, x4:],

        image[y3:, :x3],
        image[y3:, x3:x4],
        image[y4:, x4:]
    ]


def get_diagonal_lines(section):
    dst = cv.Canny(section, 50, 200, None, 3)
    lines1 = cv.HoughLines(dst, 1, np.pi / 180, 40, None, min_theta=0.6, max_theta=1)
    lines2 = cv.HoughLines(dst, 1, np.pi / 180, 40, None, min_theta=2, max_theta=2.6)
    if lines1 is not None or lines2 is not None:
        return True
    return False


def detect_circles(section):
    circles = cv.HoughCircles(section, cv.HOUGH_GRADIENT, 1, 20, param1=5, param2=30, minRadius=5, maxRadius=0)
    return circles


def classify_cells(sections):
    classes = []  # 0 - circle, 1 - x, 2 - nothing
    for sec_i in range(len(sections)):
        sec = sections[sec_i]

        canny = cv.Canny(sec, 50, 200, None, 3)
        im2, contours = cv.findContours(canny, 1, 2)
        c = max(im2, key=lambda x: len(cv.convexHull(x)))
        if len(cv.convexHull(c)) >= 30 or (len(cv.convexHull(c)) >= 20 and detect_circles(sec) is not None):
            classes.append('0')
            continue

        if get_diagonal_lines(sec):
            classes.append('X')
        else:
            classes.append('')

    return np.array(classes).reshape((3, 3))