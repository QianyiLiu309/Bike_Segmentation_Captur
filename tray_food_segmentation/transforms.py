import numpy as np
import cv2


def flip_vertically(img_data):
    return img_data[::-1, :, :]


def flip_horizontally(img_data):
    return img_data[:, ::-1, :]


def increase_brightness(img_data, value):
    hsv_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)

    if value > 0:
        upper_lim = 255 - value
        v[v > upper_lim] = 255
        v[v <= upper_lim] += value
    else:
        lower_lim = -value
        v[v < lower_lim] = 0
        v[v >= lower_lim] -= (value * -1)

    final_hsv = cv2.merge((h, s, v))
    final_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return final_rgb


def get_crop(img_data, x, y, h, w):
    x_max, y_max = img_data.shape[:2]
    if x + h > x_max or y + w > y_max:
        return None
    return img_data[x:x + h, y:y + w, :]


def projective_transformation_up(img_data, c1, c2, r1, r2):
    rows, cols = img_data.shape[:2]
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    dst_points = np.float32(
        [[0, 0], [cols - 1, 0], [int((1 - c1) * cols), int(r1 * rows)], [int(c2 * cols), int(r2 * rows)]])
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(img_data, projective_matrix, (cols, int(max(r1, r2) * rows)))
    return img_output


def projective_transformation_down(img_data, c1, c2, r1, r2):
    rows, cols = img_data.shape[:2]
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    dst_points = np.float32(
        [[int((1 - c1) * cols), int((1 - r1) * rows)], [int(c2 * cols), int((1 - r2) * rows)], [0, rows - 1],
         [cols - 1, rows - 1]])

    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(img_data, projective_matrix, (cols, rows))
    return img_output


def projective_transformation_left(img_data, c1, c2, r1, r2):
    rows, cols = img_data.shape[:2]
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    dst_points = np.float32(
        [[0, 0], [int(c1 * cols), int((1 - r1) * rows)], [0, rows - 1], [int(c2 * cols), int(r2 * rows)]])

    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(img_data, projective_matrix, (cols, rows))
    return img_output


def projective_transformation_right(img_data, c1, c2, r1, r2):
    rows, cols = img_data.shape[:2]
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    dst_points = np.float32(
        [[int((1 - c1) * cols), int((1 - r1) * rows)], [cols - 1, 0], [int((1 - c2) * cols), int(r2 * rows)],
         [cols - 1, rows - 1]])

    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(img_data, projective_matrix, (cols, rows))
    return img_output


def add_color(img_data, value):
    channels = cv2.split(img_data)
    for c, v in zip(channels, value):
        if v > 0:
            limit = 255 - v
            c[c > limit] = 255
            c[c <= limit] += v
        else:
            lower_lim = -value
            v[v < lower_lim] = 0
            v[v >= lower_lim] -= (value * -1)

    img_output = cv2.merge(channels)
    return img_output
