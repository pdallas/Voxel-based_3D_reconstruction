import shutil

import cv2 as cv
import numpy as np
import os
import glob

# chessboard size
chessboard = (8, 6)
cube_stride = 22

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Object points definition based on our chessboard
objp = np.zeros((chessboard[1] * chessboard[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

# Image points definition based on our chessboard
imgp = np.zeros((chessboard[1] * chessboard[0], 2), np.float32)
imgp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2) * 22

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points
imgpoints = []  # 2d points

# Global var that controls the click event coordinates
click_coordinates = []

"""
The following function extracts a predefined amount of images from a video source
"""


def extract_frames(cam_number, frames_to_extract):
    count = 0
    vid_cap = cv.VideoCapture(fr".\data\cam{cam_number}\intrinsics.avi")

    if not os.path.exists(f'./data/cam{cam_number}/frames/frame_0.jpg'):
        print(f"Files not found, extracting images for camera {cam_number} ..")
        while count < frames_to_extract:
            vid_cap.set(cv.CAP_PROP_POS_MSEC, (count * 5000))
            success, image = vid_cap.read()
            print('Read a new frame: ', success)
            if not os.path.exists(f'./data/cam{cam_number}/frames'):
                os.mkdir(f'./data/cam{cam_number}/frames')
            cv.imwrite('./data/cam' + str(cam_number) + '/frames/frame_' + str(count) + '.jpg', image)
            count = count + 1
    return True


"""
The following function is the implementation of the offline phase of the assignment
Firstly, the images are loaded (for each run)
Then, we try to find the Chessboard corners by using the opencv function 'findChessboardCorners'.
If the algorithm is not able to determine the corners, the user then has to provide manually all 4 chessboard corners
In the end, we calibrate the camera based on the Object points and Image points that we obtain based on the flow that is described above.
"""


def run_offline(cam_number):
    images = glob.glob('./data/cam' + str(cam_number) + '/frames/*.jpg')
    for count, fname in enumerate(images):
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboard, None)

        if ret == True:
            # Finding sub-pixel corners based on the original corners
            corners_sub_pix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_sub_pix)
            cv.drawChessboardCorners(img, chessboard, corners_sub_pix, ret)
            cv.imshow('img', img)
            #cv.imwrite('./cam_' + str(cam_number) + '/corners/cornered_img_' + str(count) + '.jpg', img)

            cv.waitKey(2000)
        else:
            # Corners not found, manually request corners
            print(f'Did not find corners for file: {fname}')
            cv.imshow('img', img)
            cv.setMouseCallback('img', click_event, img)
            cv.waitKey(0)
            #cv.imwrite('./cam_' + str(cam_number) + '/corners/cornered_img_' + str(count) + '.jpg', img)

    cv.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    save_params(cam_number, ret, mtx, dist, rvecs, tvecs)
    print(f'Intrinsic camera values for camera {cam_number} are \n\t {mtx}')


"""
The following function is used to save all the necessary parameters that the calibration function returns during the offline phase.
"""


def save_params(cam_number, ret, mtx, dist, rvecs, tvecs):
    if os.path.exists(f"./data/cam{cam_number}/params"):
        shutil.rmtree(f'./data/cam{cam_number}/params')
    os.mkdir(f'./data/cam{cam_number}/params')
    np.save(f'./data/cam{cam_number}/params/ret.npy', ret)
    np.save(f'./data/cam{cam_number}/params/mtx.npy', mtx)
    np.save(f'./data/cam{cam_number}/params/dist.npy', dist)
    np.save(f'./data/cam{cam_number}/params/rvecs.npy', rvecs)
    np.save(f'./data/cam{cam_number}/params/tvecs.npy', tvecs)



"""
The following function tracks the click events and plots the coordinates of the points clicked on the image.
"""


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, '', y)
        # displaying the coordinates on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        if len(click_coordinates) < 4:
            cv.putText(params, str(x) + ',' +
                       str(y), (x, y), font,
                       1, (255, 0, 0), 2)
            cv.imshow('img', params)
            click_coordinates.append([x, y])
        if len(click_coordinates) == 4:
            transform_image(params)


"""
The following function transforms the perspective of the image.
"""


def transform_image(img):
    new_points = np.float32(
        [[0, 0], [(chessboard[0] - 1) * 22, 0], [(chessboard[0] - 1) * 22, (chessboard[1] - 1) * 22],
         [0, (chessboard[1] - 1) * 22]])
    M = cv.getPerspectiveTransform(new_points, np.float32(click_coordinates))
    warp = cv.perspectiveTransform(np.float32(imgp)[None, :, :], M)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = np.float32(warp)
    # By using cornerSubPix we are able to find potitions of the corners more accurately
    corners_sub_pix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp)
    imgpoints.append(np.float32(corners_sub_pix))
    cv.drawChessboardCorners(img, chessboard, corners_sub_pix, True)
    cv.imshow('img', img)
    # Clear the click coordinates for each iteration
    click_coordinates.clear()
