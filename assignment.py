import glm
import random
import cv2 as cv
import numpy as np

block_size = 1.0
w = 8
h = 6
prevForeground = [None for _ in range(4)]
lookup = None
cols = [0, 0, 0, 0]


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    width = 160
    depth = 200
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0.5, 0.5, 0.5])
    return data, colors


# def get_cam_rotation_matrices():
#     # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
#     # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
#     cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
#     cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
#     for c in range(len(cam_rotations)):
#         cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
#         cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
#         cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
#     return cam_rotations

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
    # Generates random voxel locations
    global prevForeground, lookup
    width = 16
    height = 8
    depth = 16

    voxel_size = 0.2
    data0 = []
    colors = []

    for x in np.linspace(0, 16, num=100):
        for y in np.linspace(0, 16, num=100):
            for z in np.linspace(0, 16, num=100):
                data0.append([x, y, z])
    data0 = np.array(data0)

    # first frame, compute lookup table
    for i in range(4):
        print(i)
        cam_params = init_cam_params(i + 1)

        pts, _ = cv.projectPoints(data0, cam_params[2], cam_params[3], cam_params[0], cam_params[1])
       # for j in range(len(data0)):
        pts = np.int32(pts)
        flags = compare_with_fg(pts, i, cam_params[4])


    cv.destroyAllWindows()

    data = []
    columnSum = np.zeros(len(data0))


    for i in range(len(data0)):
        for j in range(len(flags)):
            columnSum[i] += flags[j][0][i][1]

    clip = cv.imread('./data/cam{}/horseman_frame.jpg'.format(2))
    # if voxels in all views are visible, show it on the screen
    for i in range(len(data0)):
        if columnSum[i] == 4:
            data.append(data0[i])
            # color.append(colorsVox[i])
            colors.append(clip[flags[1][0][i][0][0]][flags[1][0][i][0][1]] / 256)

    r_x = np.array([[1, 0, 0],
                   [0, 0, 1],
                   [0, -1, 0]])
    r_y_1 = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [-1, 0, 0]])
    r_y_2 = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [-1, 0, 0]])
    final_m = np.array([[6, 0, 0],
                   [0, 6, 0],
                   [0, 0, 6]])
    r_x_1 = [r_x.dot(p) for p in data]
    r_y_1_ = [r_y_1.dot(y) for y in r_x_1]
    r_y_2_ = [r_y_2.dot(y) for y in r_y_1_]
    final = [np.multiply(m, 3) for m in r_y_2_]

    return final, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    cam_position = []
    for i in range(4):
        fs = cv.FileStorage('./data/cam{}/config.xml'.format(i + 1), cv.FILE_STORAGE_READ)
        tvec = fs.getNode('tvec').mat()
        rvec = fs.getNode('rvec').mat()
        R, _ = cv.Rodrigues(rvec)
        R_inv = R.T
        position = -R_inv.dot(tvec)  # get camera position
        # get camera position in voxel space units(swap the y and z coordinates)
        Vposition = np.array([position[0] * 3, position[2] * 3, position[1] * 3])
        # Vposition /= 1.8
        cam_position.append(Vposition)
        color = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
    return cam_position, color


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    # cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    # cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    # for c in range(len(cam_rotations)):
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])

    cam_rotations = []
    # for i in range(4):
    #     fs = cv.FileStorage('./data/cam{}/config.xml'.format(i + 1), cv.FILE_STORAGE_READ)
    #     rvec = fs.getNode('rvec').mat()
    #     rotationMatrix = cv.Rodrigues(np.array(rvec).astype(np.float32))[0]
    #     # calculate the camera matrices
    #     rotationMatrix = rotationMatrix.transpose()
    #     rotationMatrix = [rotationMatrix[0], rotationMatrix[2], rotationMatrix[1]]
    #     cam_rotations.append(glm.mat4(np.matrix(rotationMatrix).T))
    # print("F:")
    # print(cam_rotations)
    #
    # for c in range(len(cam_rotations)):
    #     # transform from radians to degrees.
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], -np.pi / 2, [0, 1, 0])
    for i in range(4):
        fs = cv.FileStorage('./data/cam{}/config.xml'.format(i + 1), cv.FILE_STORAGE_READ)
        rvec = fs.getNode('rvec').mat()
        R, _ = cv.Rodrigues(rvec)

        R[:, 1], R[:, 2] = R[:, 2], R[:, 1].copy()  # swap y and z (exchange the second and third columns)
        R[1, :] = -R[1, :]  # invert rotation on y (multiply second row by -1)
        # rotation matrix: rotation 90 degree about the y
        rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        R = np.matmul(R, rot)

        # convert to mat4x4 format
        RM = np.eye(4)
        RM[:3, :3] = R
        RM = glm.mat4(*RM.flatten())
        cam_rotations.append(RM)
    # print(cam_rotations)
    return cam_rotations
