import cv2 as cv
from intrinsics import extract_frames, run_offline, getExtrinsics, manual_corners, background_model, background_sub

cam_number = 1
frames_to_extract = 10


def offline():
    #extract_frames(cam_number, frames_to_extract)
    #run_offline(cam_number)
    #getExtrinsics(cam_number)
    #manual_corners(cam_number)
    #background_model(cam_number)
    background_sub(cam_number)


if __name__ == "__main__":
    offline()
