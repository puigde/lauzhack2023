import os
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
import argparse
import torch.nn as nn
from modules import resnet50

import cv2 as cv

SAVED_CALIBRATIONS_PATH = "./saved_calibrations/"
CAMERA_CALIBRATIONS_PATH = os.path.join(SAVED_CALIBRATIONS_PATH, "camera")
FACE_CALIBRATIONS_PATH = os.path.join(SAVED_CALIBRATIONS_PATH, "face")

trans = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv.solvePnP(
        face_model, landmarks, camera, distortion, flags=cv.SOLVEPNP_EPNP
    )

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv.solvePnP(
            face_model, landmarks, camera, distortion, rvec, tvec, True
        )

    return rvec, tvec


def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = np.min([h, w]) / 2.0
    pos = (int(w / 2.0), int(h / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv.arrowedLine(
        image_out,
        tuple(np.round(pos).astype(np.int32)),
        tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)),
        color,
        thickness,
        cv.LINE_AA,
        tipLength=0.2,
    )

    return image_out, dx, dy


def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(
        np.concatenate((two_eye_center, nose_center), axis=1), axis=1
    ).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(
        face_center
    )  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array(
        [  # camera intrinsic parameters of the virtual camera
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ]
    )
    S = np.array(
        [  # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ]
    )

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(
        np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))
    )  # transformation matrix

    img_warped = cv.warpPerspective(img, W, roiSize)  # warp the input image

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize the facial landmarks
    num_point = landmarks.shape[0]
    landmarks_warped = cv.perspectiveTransform(landmarks, W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)

    return img_warped, landmarks_warped


def pipeline_single_image(
    image,
    predictor,
    face_detector,
    model,
    cam_file_name=f"{CAMERA_CALIBRATIONS_PATH}/camera.xml",
):
    detected_faces = face_detector(
        cv.cvtColor(image, cv.COLOR_BGR2RGB), 1
    )  ## convert BGR image to RGB for dlib
    if len(detected_faces) == 0:
        print("warning: no detected face")
        exit(0)
    print("detected one face")
    shape = predictor(
        image, detected_faces[0]
    )  ## only use the first detected face (assume that each input image only contains one face)
    shape = face_utils.shape_to_np(shape)
    landmarks = []
    for (x, y) in shape:
        landmarks.append((x, y))
    landmarks = np.asarray(landmarks)

    # load camera information
    if not os.path.isfile(cam_file_name):
        print("no camera calibration file is found.")
        exit(0)
    fs = cv.FileStorage(cam_file_name, cv.FILE_STORAGE_READ)
    camera_matrix = fs.getNode(
        "Camera_Matrix"
    ).mat()  # camera calibration information is used for data normalization
    camera_distortion = fs.getNode("Distortion_Coefficients").mat()

    print("estimate head pose")
    face_model_load = np.loadtxt(
        f"{FACE_CALIBRATIONS_PATH}/face_model.txt"
    )  # Generic face model with 3D facial landmarks
    landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    face_model = face_model_load[landmark_use, :]
    # estimate the head pose,
    ## the complex way to get head pose information, eos library is required,  probably more accurrated
    # landmarks = landmarks.reshape(-1, 2)
    # head_pose_estimator = HeadPoseEstimator()
    # hr, ht, o_l, o_r, _ = head_pose_estimator(image, landmarks, camera_matrix[cam_id])
    ## the easy way to get head pose information, fast and simple
    facePts = face_model.reshape(6, 1, 3)
    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
    landmarks_sub = landmarks_sub.astype(
        float
    )  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(
        6, 1, 2
    )  # input to solvePnP requires such shape
    hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)

    # data normalization method
    print("data normalization, i.e. crop the face image")
    img_normalized, landmarks_normalized = normalizeData_face(
        image, face_model, landmarks_sub, hr, ht, camera_matrix
    )

    input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
    input_var = trans(input_var)
    input_var = torch.autograd.Variable(input_var.float())
    input_var = input_var.view(
        1, input_var.size(0), input_var.size(1), input_var.size(2)
    )  # the input must be 4-dimension
    pred_gaze = model(
        input_var
    )  # get the output gaze direction, this is 2D output as pitch and raw rotation
    pred_gaze = pred_gaze[
        0
    ]  # here we assume there is only one face inside the image, then the first one is the prediction
    pred_gaze_np = (
        pred_gaze.cpu().data.numpy()
    )  # convert the pytorch tensor to numpy array
    return img_normalized, landmarks_normalized, pred_gaze_np


class gaze_network(nn.Module):
    def __init__(self, use_face=False, num_glimpses=1):
        super(gaze_network, self).__init__()
        self.gaze_network = resnet50(pretrained=True)

        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        gaze = self.gaze_fc(feature)

        return gaze
