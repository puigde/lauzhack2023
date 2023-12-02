import numpy as np
import pickle
from lxml import etree
from lxml.builder import E
from subprocess import call
import cv2
import numpy as np
from os import path
import argparse
import os

SAVED_CALIBRATIONS_PATH = "./saved_calibrations/"
CAMERA_CALIBRATIONS_PATH = os.path.join(SAVED_CALIBRATIONS_PATH, "camera")


def cam_calibrate(cam_idx, cap, cam_calib):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    pts = np.zeros((6 * 9, 3), np.float32)
    pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # capture calibration frames
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    frames = []
    while True:
        ret, frame = cap.read()
        frame_copy = frame.copy()

        corners = []
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            retc, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if retc:
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # Draw and display the corners
                cv2.drawChessboardCorners(frame_copy, (9, 6), corners, ret)

                cv2.imshow("points", frame_copy)
                # s to save, c to continue, q to quit
                if cv2.waitKey(0) & 0xFF == ord("s"):
                    img_points.append(corners)
                    obj_points.append(pts)
                    frames.append(frame)
                elif cv2.waitKey(0) & 0xFF == ord("c"):
                    continue
                elif cv2.waitKey(0) & 0xFF == ord("q"):
                    print("Calibrating camera...")
                    cv2.destroyAllWindows()
                    break

    # compute calibration matrices

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, frames[0].shape[0:2], None, None
    )

    # check
    error = 0.0
    for i in range(len(frames)):
        proj_imgpoints, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], mtx, dist
        )
        error += cv2.norm(img_points[i], proj_imgpoints, cv2.NORM_L2) / len(
            proj_imgpoints
        )
    print(
        "Camera calibrated successfully, total re-projection error: %f"
        % (error / len(frames))
    )

    cam_calib["mtx"] = mtx
    cam_calib["dist"] = dist
    print("Camera parameters:")
    print(cam_calib)

    print(mtx, dist)

    xml_content = f"""<?xml version="1.0"?>
        <opencv_storage>
        <Camera_Matrix type_id="opencv-matrix">
        <rows>3</rows>
        <cols>3</cols>
        <dt>d</dt>
        <data>
            {' '.join(map(str, mtx.flatten()))}
        </data></Camera_Matrix>
        <Distortion_Coefficients type_id="opencv-matrix">
        <rows>1</rows>
        <cols>{dist.size}</cols>
        <dt>d</dt>
        <data>
            {' '.join(map(str, dist.flatten()))}
        </data></Distortion_Coefficients>
        </opencv_storage>"""

    fname = (
        f"{CAMERA_CALIBRATIONS_PATH}/camera.xml"
        if cam_idx == 2
        else f"{CAMERA_CALIBRATIONS_PATH}/camera_pocha.xml"
    )
    with open(fname, "w") as f:
        f.write(xml_content)


#################################
# Start camera
#################################

parser = argparse.ArgumentParser()
parser.add_argument("--cam_id", type=int, default=0)
args = parser.parse_args()

cam_idx = args.cam_id


# adjust these for your camera to get the best accuracy
# use the same parameters to run the actual demoqsqq
call("v4l2-ctl -d /dev/video%d -c brightness=100" % cam_idx, shell=True)
call("v4l2-ctl -d /dev/video%d -c contrast=50" % cam_idx, shell=True)
call("v4l2-ctl -d /dev/video%d -c sharpness=100" % cam_idx, shell=True)

cam_cap = cv2.VideoCapture(cam_idx)
cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# calibrate camera
cam_calib = {"mtx": np.eye(3), "dist": np.zeros((1, 5))}
print(
    "Calibrate camera once. Print pattern.png, paste on a clipboard, show to camera and capture non-blurry images in which points are detected well."
)
print(
    "Press s to save frame, c to continue to next frame and q to quit collecting data and proceed to calibration."
)
cam_calibrate(cam_idx, cam_cap, cam_calib)
cam_cap.release()
