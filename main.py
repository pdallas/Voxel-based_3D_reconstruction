import cv2 as cv
from intrinsics import extract_frames, run_offline

cam_number = 3
frames_to_extract = 10


def offline():
    extract_frames(cam_number, frames_to_extract)
    run_offline(cam_number)


if __name__ == "__main__":
    offline()
