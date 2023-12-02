import gi.repository

gi.require_version("Gdk", "3.0")
from gi.repository import Gdk
import random
import threading
import pickle
import os
import sys
import cv2 as cv
import dlib
import torch
import numpy as np
import pickle
from subprocess import call
from model import gaze_network
from utils import draw_gaze, pipeline_single_image

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

    font = cv.FONT_HERSHEY_SIMPLEX
    img = np.ones((h, w, 3), np.float32)
    if direction == "r" or direction == "l":
        if direction == "r":
            cv.putText(img, target, (x, y), font, 0.5, color, 2, cv.LINE_AA)
        elif direction == "l":
            cv.putText(img, target, (w - x, y), font, 0.5, color, 2, cv.LINE_AA)
            img = cv.flip(img, 1)
    elif direction == "u" or direction == "d":
        imgT = np.ones((w, h, 3), np.float32)
        if direction == "d":
            cv.putText(imgT, target, (y, x), font, 0.5, color, 2, cv.LINE_AA)
        elif direction == "u":
            cv.putText(imgT, target, (h - y, x), font, 0.5, color, 2, cv.LINE_AA)
            imgT = cv.flip(imgT, 1)
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

    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.setWindowProperty("image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    calib_data = {"frames": [], "g_t": []}

    i = 0
    while i < calib_points:

        # Start the sub-thread, which is responsible for grabbing images
        frames = []
        THREAD_RUNNING = True
        th = threading.Thread(target=grab_img, args=(cap,))
        
        direction = random.choice(directions)
        img, g_t = create_image(
            mon, direction, i, (0, 0, 0), grid=True, total=calib_points
        )
        cv.imshow("image", img)
        key_press = cv.waitKey(0)
        th.start()
        if key_press == keys[direction]:
            THREAD_RUNNING = False
            th.join()
            calib_data["frames"].append(frames)
            calib_data["g_t"].append(g_t)
            i += 1
        elif key_press & 0xFF == ord("q"):
            cv.destroyAllWindows()
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
        
        direction = random.choice(directions)
        img, g_t = create_image(
            mon, direction, i, (0, 0, 0), grid=False, total=rand_points
        )
        cv.imshow("image", img)
        key_press = cv.waitKey(0)
        th.start()
        if key_press == keys[direction]:
            THREAD_RUNNING = False
            th.join()
            calib_data["frames"].append(frames)
            calib_data["g_t"].append(g_t)
            i += 1
        elif key_press & 0xFF == ord("q"):
            cv.destroyAllWindows()
            break
        else:
            THREAD_RUNNING = False
            th.join()
    cv.destroyAllWindows()

    return calib_data

def main():
    cam_idx = 0
    cam_cap = cv.VideoCapture(cam_idx)
    cam_cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cam_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    mon = monitor()
    '''
    load models and shit
    '''
    predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
    #face_detector = dlib.cnn_face_detection_model_v1('./modules/mmod_human_face_detector.dat')
    face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful
    print('load gaze estimator')
    model = gaze_network()
    #model.cuda() # comment this line out if you are not using GPU
    pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
    if not os.path.isfile(pre_trained_model_path):
        print('the pre-trained gaze estimation model does not exist.')
        exit(0)
    else:
        print('load the pre-trained model: ', pre_trained_model_path)
    ckpt = torch.load(pre_trained_model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
    model.eval()  # change it to the evaluation mode
    data = collect_data(cam_cap, mon, calib_points=9, rand_points=4)
    max_x, max_y, min_x, min_y = -float('inf'),-float('inf'),float('inf'),float('inf')
    for im in data['frames']:
        print(np.array(im).shape)
        img_normalized, landmarks_normalized, pred_gaze_np =  pipeline_single_image(im[0], predictor, face_detector, model)
        _,x,y = draw_gaze(img_normalized, pred_gaze_np)
        max_x=max(max_x,x)
        max_y=max(max_y,y)
        min_x=min(min_x,x)
        min_y=min(min_y,y)
    
    with open('screen_corners.pkl','wb') as f:
        pickle.dump((min_x,min_y,max_x,max_y),f)

if __name__ == "__main__":
    main()
