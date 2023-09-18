# Py version: 3.9.0
# Author gmail: kriuchkvskyi.vlad@gmail.com

# Imports
import cv2 as cv
import os

from draw import draw_winner
from detection import *

INPUT_DIR = 'images'
OUTPUT_DIR = 'results'

if __name__ == '__main__':
    INPUT_DIR = 'images'
    for filename in os.listdir(INPUT_DIR):
        img_path = os.path.join(INPUT_DIR, filename)

        img = cv.imread(img_path, 0)

        thresholded_image = apply_threshold(img)
        cropped_image = crop_img_f(thresholded_image)

        dilation = dilate_image(cropped_image)

        dx, dy = get_derievetives(dilation)
        vertical_lines, horizontal_lines = get_vertical_lines(dy), get_horizontal_lines(dx)

        intersection_points = get_intersections(horizontal_lines, vertical_lines)

        sections_thresholded = get_sections(thresholded_image, intersection_points)

        classified = classify_cells(sections_thresholded)
        cdist = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cdist = draw_winner(classified, cdist, intersection_points, sections_thresholded)

        cv.imwrite(os.path.join(OUTPUT_DIR, filename), cdist)
