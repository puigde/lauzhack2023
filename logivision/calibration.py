import gi.repository

gi.require_version("Gdk", "3.0")
from gi.repository import Gdk
import numpy as np
import cv2
import numpy as np
import random
import threading
import pickle
import sys
import cv2
import numpy as np
import pickle
from subprocess import call

sys.path.append("../logvision/")

directions = ["l", "r", "u", "d"]
keys = {"u": 82, "d": 84, "l": 81, "r": 83}

global THREAD_RUNNING
global frames


class monitor:
    def __init__(self):
        display = Gdk.Display.get_default()
        screen = display.get_default_screen()
        default_screen = screen.get_default()
        num = default_screen.get_number()

        self.h_mm = default_screen.get_monitor_height_mm(num)
        self.w_mm = default_screen.get_monitor_width_mm(num)

        self.h_pixels = default_screen.get_height()
        self.w_pixels = default_screen.get_width()

    def monitor_to_camera(self, x_pixel, y_pixel):

        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        x_cam_mm = ((int(self.w_pixels / 2) - x_pixel) / self.w_pixels) * self.w_mm
        y_cam_mm = 10.0 + (y_pixel / self.h_pixels) * self.h_mm
        z_cam_mm = 0.0

        return x_cam_mm, y_cam_mm, z_cam_mm

    def camera_to_monitor(self, x_cam_mm, y_cam_mm):
        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        x_mon_pixel = np.ceil(
            int(self.w_pixels / 2) - x_cam_mm * self.w_pixels / self.w_mm
        )
        y_mon_pixel = np.ceil((y_cam_mm - 10.0) * self.h_pixels / self.h_mm)

        return x_mon_pixel, y_mon_pixel


def create_image(mon, direction, i, color, target="E", grid=True, total=9):

    h = mon.h_pixels
    w = mon.w_pixels
    if grid:
        if total == 9:
            row = i % 3
            col = int(i / 3)
            x = int((0.02 + 0.48 * row) * w)
            y = int((0.02 + 0.48 * col) * h)
        elif total == 16:
            row = i % 4
            col = int(i / 4)
            x = int((0.05 + 0.3 * row) * w)
            y = int((0.05 + 0.3 * col) * h)
    else:
        x = int(random.uniform(0, 1) * w)
        y = int(random.uniform(0, 1) * h)

    # compute the ground truth point of regard
    x_cam, y_cam, z_cam = mon.monitor_to_camera(x, y)
    g_t = (x_cam, y_cam)

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = np.ones((h, w, 3), np.float32)
    if direction == "r" or direction == "l":
        if direction == "r":
            cv2.putText(img, target, (x, y), font, 0.5, color, 2, cv2.LINE_AA)
        elif direction == "l":
            cv2.putText(img, target, (w - x, y), font, 0.5, color, 2, cv2.LINE_AA)
            img = cv2.flip(img, 1)
    elif direction == "u" or direction == "d":
        imgT = np.ones((w, h, 3), np.float32)
        if direction == "d":
            cv2.putText(imgT, target, (y, x), font, 0.5, color, 2, cv2.LINE_AA)
        elif direction == "u":
            cv2.putText(imgT, target, (h - y, x), font, 0.5, color, 2, cv2.LINE_AA)
            imgT = cv2.flip(imgT, 1)
        img = imgT.transpose((1, 0, 2))

    return img, g_t


def grab_img(cap):
    global THREAD_RUNNING
    global frames
    while THREAD_RUNNING:
        _, frame = cap.read()
        frames.append(frame)


def collect_data(cap, mon, calib_points=9, rand_points=5):
    global THREAD_RUNNING
    global frames

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    calib_data = {"frames": [], "g_t": []}

    i = 0
    while i < calib_points:

        # Start the sub-thread, which is responsible for grabbing images
        frames = []
        THREAD_RUNNING = True
        th = threading.Thread(target=grab_img, args=(cap,))
        th.start()
        direction = random.choice(directions)
        img, g_t = create_image(
            mon, direction, i, (0, 0, 0), grid=True, total=calib_points
        )
        cv2.imshow("image", img)
        key_press = cv2.waitKey(0)
        if key_press == keys[direction]:
            THREAD_RUNNING = False
            th.join()
            calib_data["frames"].append(frames)
            calib_data["g_t"].append(g_t)
            i += 1
        elif key_press & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        else:
            THREAD_RUNNING = False
            th.join()

    i = 0
    while i < rand_points:

        # Start the sub-thread, which is responsible for grabbing images
        frames = []
        THREAD_RUNNING = True
        th = threading.Thread(target=grab_img, args=(cap,))
        th.start()
        direction = random.choice(directions)
        img, g_t = create_image(
            mon, direction, i, (0, 0, 0), grid=False, total=rand_points
        )
        cv2.imshow("image", img)
        key_press = cv2.waitKey(0)
        if key_press == keys[direction]:
            THREAD_RUNNING = False
            th.join()
            calib_data["frames"].append(frames)
            calib_data["g_t"].append(g_t)
            i += 1
        elif key_press & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        else:
            THREAD_RUNNING = False
            th.join()
    cv2.destroyAllWindows()

    return calib_data


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

    pickle.dump(cam_calib, open("calib_cam%d.pkl" % (cam_idx), "wb"))


def main():
    cam_idx = 0
    cam_cap = cv2.VideoCapture(cam_idx)
    cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    mon = monitor()
    # data = collect_data(cam_cap, mon, calib_points=9, rand_points=4)


if __name__ == "__main__":
    main()
