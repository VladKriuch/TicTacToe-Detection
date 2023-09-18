import numpy as np
import cv2 as cv


def check_rows(classes):
    for row_index in range(len(classes)):
        if np.all(classes[row_index] == classes[row_index][0]):
            return row_index
    return None


def check_cols(classes):
    for column_index in range(len(classes[0])):
        if np.all(classes[:, column_index] == classes[:, column_index][0]):
            return column_index
    return None


def check_diags(classes):
    main_diagonal1 = np.diagonal(classes)
    main_diagonal2 = np.diagonal(np.fliplr(classes))
    if np.all(main_diagonal1 == main_diagonal1[0]):
        return 0
    elif np.all(main_diagonal2 == main_diagonal2[0]):
        return 1
    return None


def draw_winner(classes, img, intersection_points, sections):
    intersection_points = sorted(intersection_points, key=lambda p: p[0])

    p1, p3 = sorted(intersection_points[:2], key=lambda p: p[1])
    p2, p4 = sorted(intersection_points[2:], key=lambda p: p[1])
    if check_cols(classes):
        index = check_cols(classes)
        start_index = index

        end_index = start_index + 6

        section_start = sections[start_index]
        im2, contours = cv.findContours(section_start, 1, 2)
        c = max(im2, key=lambda x: len(x))

        M = cv.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if start_index == 1:
            cx += p1[0]
        if start_index == 2:
            cx += p2[0]
        cy -= int(cy / 2)
        point_start = [cx, cy]

        section_end = sections[end_index]
        im2, contours = cv.findContours(section_end, 1, 2)
        c = max(im2, key=cv.contourArea)

        M = cv.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if start_index == 0:
            cy += p3[1]
        if start_index == 1:
            cy += p3[1]
            cx += p3[0]
        if start_index == 2:
            cy += p4[1]
            cx += p4[0]

        point_end = [cx, cy]
        cv.line(img, point_start, point_end, (0, 0, 255), 3, cv.LINE_AA)

    if check_rows(classes):
        index = check_rows(classes)
        if index == 0:
            start_index = 0
        elif index == 1:
            start_index = 3
        elif index == 2:
            start_index = 6

        end_index = start_index + 2

        section_start = sections[start_index]
        im2, contours = cv.findContours(section_start, 1, 2)
        c = max(im2, key=cv.contourArea)

        M = cv.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if start_index == 3:
            cy += p1[1]
        if start_index == 6:
            cy += p3[1]

        point_start = [cx, cy]

        section_end = sections[end_index]
        im2, contours = cv.findContours(section_end, 1, 2)
        c = max(im2, key=lambda x: len(x))

        M = cv.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if start_index == 0:
            cx += p2[0]
        if start_index == 3:
            cy += p1[1]
            cx += p2[0]
        if start_index == 6:
            cy += p4[1]
            cx += p4[0]

        point_end = [cx, cy]
        cv.line(img, point_start, point_end, (0, 0, 255), 3, cv.LINE_AA)

    if check_diags(classes):
        index = check_diags(classes)
        if index == 0:
            start_index = 0
        elif index == 1:
            start_index = 2

        if index == 0:
            end_index = 8
        elif index == 1:
            end_index = 6

        section_start = sections[start_index]
        im2, contours = cv.findContours(section_start, 1, 2)
        c = max(im2, key=lambda x: len(x))

        M = cv.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if start_index == 2:
            cx += p4[0]
        point_start = [cx, cy]

        section_end = sections[end_index]
        im2, contours = cv.findContours(section_end, 1, 2)
        c = max(im2, key=cv.contourArea)

        M = cv.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if start_index == 0:
            cx += p4[0]
            cy += p4[1]
        if start_index == 2:
            cy += p3[1]

        point_end = [cx, cy]
        cv.line(img, point_start, point_end, (0, 0, 255), 3, cv.LINE_AA)

    return img