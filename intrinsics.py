import shutil

from sklearn.mixture import GaussianMixture as GMM
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


def extrinsics(cam_number):

    data = np.load(f'data/cam{cam_number}/sub_pix.npy')
    corners = np.array(data)

    fs = cv.FileStorage(f'./data/cam{cam_number}/intrinsics.xml', cv.FILE_STORAGE_READ)
    mtx = fs.getNode('mtx').mat()
    dist = fs.getNode('dist').mat()


    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    # extract the extrinsic parameters
    ret, rvec, tvec = cv.solvePnP(objp, corners, mtx, dist)
    R, _ = cv.Rodrigues(rvec)  # change rotation vector to matrix
    T, _ = cv.Rodrigues(tvec)  # change translation vector to matrix

    fs = cv.FileStorage(f'./data/cam{cam_number}/config.xml', cv.FILE_STORAGE_WRITE)
    fs.write("mtx", mtx)
    fs.write("dist", dist)
    fs.write("rvec", rvec)
    fs.write("tvec", tvec)


    img = cv.imread(f'./data/cam{cam_number}/board.jpg')
    pts = np.float32([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]])
    image_pts, _ = cv.projectPoints(pts, rvec, tvec, mtx, dist)

    image_pts = np.int32(image_pts).reshape(-1, 2)

    img = cv.line(img, tuple(image_pts[0]), tuple(image_pts[1]), (0, 0, 255), 2)
    img = cv.line(img, tuple(image_pts[0]), tuple(image_pts[2]), (0, 255, 0), 2)
    img = cv.line(img, tuple(image_pts[0]), tuple(image_pts[3]), (255, 0, 0), 2)
    cv.imwrite(f'./data/cam{cam_number}/draw_axes.jpg', img)
    cv.imshow('axes', img)
    cv.waitKey(10000)
    cv.destroyAllWindows()


"""
The following function extracts a predefined amount of images from a video source
"""


def extract_frames(cam_number, frames_to_extract, filename):
    count = 0
    vid_cap = cv.VideoCapture(fr".\data\cam{cam_number}\{filename}.avi")

    if os.path.exists(f'./data/cam{cam_number}/frames/frame_0.jpg'):
        print(f"Files not found, extracting images for camera {cam_number} ..")
        while count < frames_to_extract:
            vid_cap.set(cv.CAP_PROP_POS_MSEC, (count * 200))
            success, image = vid_cap.read()
            print('Read a new frame: ', success)
            if not os.path.exists(f'./data/cam{cam_number}/frames'):
                os.mkdir(f'./data/cam{cam_number}/frames')
            cv.imwrite('./data/cam' + str(cam_number) + '/background_frames/' + str(count) + '.jpg', image)
            count = count + 1
    return True


def manual_corners(cam_number):
    images = glob.glob('./data/cam' + str(cam_number) + '/board.jpg')
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
            cv.waitKey(2000)
        else:
            # Corners not found, manually request corners
            print(f'Did not find corners for file: {fname}')
            cv.imshow('img', img)
            cv.setMouseCallback('img', click_event, img)
            cv.waitKey(0)

    cv.destroyAllWindows()


# TODO; Fix GMM model for the background frame selection
def background_model(camera_num):
    total_frames = 0
    for count, frame in enumerate(glob.glob(f"./data/cam{camera_num}/background_frames/*.jpg")):
        vid_frame = cv.imread(frame)
        total_frames = total_frames + vid_frame.astype('float')
    average_frame = total_frames / count
    cv.imwrite(f"./data/cam{camera_num}/background_avg.jpg", average_frame)
    cv.waitKey(0)
    #     #img = cv.imread(f"./data/cam{camera_num}/background_0.jpg")
    #     img = cv.imread(image)
    #     #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     #print(gray.shape)
    #     print(img.shape)
    #     #Convert MxNx3 image into Kx3 where K=MxN
    #     img2 = img.reshape((-1, 3))  # -1 reshape means, in this case MxN
    #     print(img2.shape)
    #     # covariance choices, full, tied, diag, spherical
    #     gmm_model = GMM(n_components=2, covariance_type='full').fit(img2)  # tied works better than full
    # print(gmm_model.means_)
    # gmm_labels = gmm_model.predict(img2)
    # sample = gmm_model.sample(1)
    # print(sample)

    # # Put numbers back to original shape so we can reconstruct segmented image
    # original_shape = img.shape
    # segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
    # #gray = cv.cvtColor(segmented, cv.COLOR_BGR2GRAY)
    # #cv.imwrite(f"./data/cam{camera_num}/background_0_segtest.jpg", sample)


def background_sub(camera_num):
    vid_cap = cv.VideoCapture(fr".\data\cam{camera_num}\video.avi")
    bg_ = cv.imread(f"./data/cam{camera_num}/background_avg.jpg")

    bg_static = cv.cvtColor(bg_, cv.COLOR_BGR2HSV)
    while True:
        ret, frame = vid_cap.read()
        if not ret:
            break
        current_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        fg = np.abs(cv.subtract(bg_static, current_frame))

        # Split the forground into three channels H, S, V
        h, s, v = cv.split(fg)

        # Apply Gaussian blur for noise reduction
        blur_v = cv.GaussianBlur(v, (5, 5), 0)
        blur_s = cv.GaussianBlur(s, (5, 5), 0)
        blur_h = cv.GaussianBlur(h, (5, 5), 0)

        # Threshold OTSU
        retv, th_v = cv.threshold(blur_v, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        rets, th_s = cv.threshold(blur_s, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        reth, th_h = cv.threshold(blur_h, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Adaptive mean thresholding
        # th_v = cv.adaptiveThreshold(blur_v, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        # th_s = cv.adaptiveThreshold(blur_s, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        # th_h = cv.adaptiveThreshold(blur_h, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

        #Adaptive gaussian thresholding
        # th_v = cv.adaptiveThreshold(blur_v, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        # th_s = cv.adaptiveThreshold(blur_s, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        # th_h = cv.adaptiveThreshold(blur_h, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

        # Channel V
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
        morph_v = cv.morphologyEx(th_v, cv.MORPH_OPEN, kernel)
        morph_v = cv.dilate(morph_v, kernel, iterations=5)
        result_v = cv.bitwise_and(morph_v, morph_v, mask=th_v)


        # Channel H
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6))
        morph_h = cv.morphologyEx(th_h, cv.MORPH_OPEN, kernel)
        morph_h = cv.dilate(morph_h, kernel, iterations=5)
        result_h = cv.bitwise_and(morph_h, morph_h, mask=th_h)

        # Channel S
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6))
        morph_s = cv.morphologyEx(th_s, cv.MORPH_OPEN, kernel)
        morph_s = cv.dilate(morph_s, kernel, iterations=5)
        result_s = cv.bitwise_and(morph_s, morph_s, mask=th_s)

        # Mask combination
        fg_s_v = cv.bitwise_or(result_s, result_h)
        fg_mask = cv.bitwise_or(result_v, fg_s_v)

        res = cv.bitwise_and(frame, frame, mask=fg_mask)
        cv.imwrite(f'./data/cam{camera_num}/foreground.jpg', fg_mask)
        cv.imwrite(f'./data/cam{camera_num}/horseman_frame.jpg', res)


        #rec_img_dilated = cv.merge([result_h, result_s, result_v])
        #bgr_img = cv.cvtColor(rec_img_dilated, cv.COLOR_HSV2BGR)

        cv.imshow('Current frame with forground mask applied', res)
        cv.imshow('Forground mask', fg_mask)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    vid_cap.release()
    cv.destroyAllWindows()


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

            cv.waitKey(2000)
        # else:
        #     # Corners not found, manually request corners
        #     print(f'Did not find corners for file: {fname}')
        #     cv.imshow('img', img)
        #     cv.setMouseCallback('img', click_event, img)
        #     cv.waitKey(0)

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

    fs = cv.FileStorage(f'./data/cam{cam_number}/intrinsics.xml', cv.FILE_STORAGE_WRITE)
    fs.write("mtx", mtx)
    fs.write("dist", dist)


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
                       0.2, (255, 0, 0), 1)
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
    # By using cornerSubPix we are able to find positions of the corners more accurately
    corners_sub_pix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    sub_pix = np.array(corners_sub_pix, np.float32)
    np.save(f'data/cam4/sub_pix.npy', sub_pix)

    objpoints.append(objp)
    imgpoints.append(np.float32(corners_sub_pix))
    cv.drawChessboardCorners(img, chessboard, corners_sub_pix, True)
    cv.imshow('img', img)
    # Clear the click coordinates for each iteration
    click_coordinates.clear()
