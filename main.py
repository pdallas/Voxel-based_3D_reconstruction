from intrinsics import extract_frames, run_offline, extrinsics, manual_corners, background_model, background_sub

cam_number = 4
frames_to_extract = 10
filename = 'background'

def run():
    #extract_frames(cam_number, frames_to_extract, filename)
    #run_offline(cam_number)
    #manual_corners(cam_number) #change dir for sub_pix.npy
    extrinsics(cam_number)
    #background_model(cam_number)
    #background_sub(cam_number)


if __name__ == "__main__":
    run()
