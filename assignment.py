import glm
import random
import cv2 as cv
import numpy as np

block_size = 1.0
cols = [0, 0, 0, 0]


def generate_grid(width, depth):
    # Generates the floor grid locations
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0.5, 0.5, 0.5])
    return data, colors


def init_cam_params(cam_no):
    fg = cv.imread(f'./data/cam{cam_no}/foreground.jpg')
    config = cv.FileStorage(f'./data/cam{cam_no}/config.xml', cv.FILE_STORAGE_READ)
    mtx = config.getNode('mtx').mat()
    dist = config.getNode('dist').mat()
    rvec = config.getNode('rvec').mat()
    tvec = config.getNode('tvec').mat()
    return mtx, dist, rvec, tvec, fg


def compare_with_fg(pts, i, fg):
    array = []
    for _ in range(len(pts)):
        array.append([[], []])
    final_array = [array]

    for y in range(len(pts)):
        value = 0
        for k in fg[pts[y][0][1]][pts[y][0][0]]:
            value += k

        if value > 0:
            final_array[0][y] = [[pts[y][0][1], pts[y][0][0]], True]
        elif value == 0:
            final_array[0][y] = [[pts[y][0][1], pts[y][0][0]], False]
        else:
            print('The value is non valid.')
            break

    cols[i] = final_array
    return cols


def set_voxel_positions(width, height, depth):
    data = []
    colors = []
    final_data = []
    frame = cv.imread('./data/cam2/video.jpg')

    for x in np.linspace(0, 16, num=100):
        for y in np.linspace(0, 16, num=100):
            for z in np.linspace(0, 16, num=100):
                data.append([x, y, z])
    data = np.array(data)

    for i in range(4):
        cam_params = init_cam_params(i + 1)
        pts, _ = cv.projectPoints(data, cam_params[2], cam_params[3], cam_params[0], cam_params[1])
        pts = np.int32(pts)
        values = compare_with_fg(pts, i, cam_params[4])

    final_values = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(len(values)):
            final_values[i] += values[j][0][i][1]

    for i in range(len(data)):
        if final_values[i] == 4:
            final_data.append(data[i])
            colors.append(frame[values[1][0][i][0][0]][values[1][0][i][0][1]] / 256)

    r_x = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]])
    r_y_1 = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [-1, 0, 0]])
    r_y_2 = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [-1, 0, 0]])
    r_x_1 = [r_x.dot(p) for p in final_data]
    r_y_1_ = [r_y_1.dot(y) for y in r_x_1]
    r_y_2_ = [r_y_2.dot(y) for y in r_y_1_]
    final = [np.multiply(m, 3) for m in r_y_2_]

    return final, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    cam_position = []
    for i in range(4):
        fs = cv.FileStorage(f'./data/cam{i+1}/config.xml', cv.FILE_STORAGE_READ)
        rotation_matrix, _ = cv.Rodrigues(fs.getNode('rvec').mat())
        camera_position = -rotation_matrix.T.dot(fs.getNode('tvec').mat())
        final_position = [camera_position[0] * 4, camera_position[2] * 4, camera_position[1] * 4]
        cam_position.append(final_position)
        color = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
    return cam_position, color


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    cam_rotations = []
    for i in range(4):
        fs = cv.FileStorage(f'./data/cam{i+1}/config.xml', cv.FILE_STORAGE_READ)
        rotation_matrix, _ = cv.Rodrigues(fs.getNode('rvec').mat())
        rotation_matrix_1 = rotation_matrix[:, [0, 2, 1]]
        rotation_matrix_1[1, :] *= -1
        final_rotation_matrix = np.eye(4)
        final_rotation_matrix[:3, :3] = np.matmul(rotation_matrix_1, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
        cam_rotations.append(glm.mat4(*final_rotation_matrix.flatten()))

    return cam_rotations
