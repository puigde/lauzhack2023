import os
import dlib
import pickle
import torch
from model import gaze_network
import argparse

import cv2 as cv

from utils import pipeline_single_image, draw_gaze




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_id', type = int, default = 0)
    args = parser.parse_args()
    '''
    load models and shit
    '''
    predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
    # face_detector = dlib.cnn_face_detection_model_v1('./modules/mmod_human_face_detector.dat')
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

    '''
    load screen corner values
    '''
    with open('screen_corners.pkl','rb') as f:
        min_x,min_y,max_x,max_y = pickle.load(f)
    print(min_x,min_y,max_x,max_y)

    # if os.path.isdir(args.input):
    #     names = os.listdir(args.input)
    #     inlist = [os.path.join(args.input,name) for name in names]
    #     outlist = [os.path.join(args.output,name) for name in names]
    #     os.makedirs(args.output, exist_ok = True)
    # else:
    #     inlist = [args.input]
    #     outlist = [args.output]
    
    # for i,o in zip(inlist,outlist):
    #     print('load input face image: ', i)
    #     image = cv2.imread(i)
    #     img_normalized, landmarks_normalized, pred_gaze_np =  pipeline_single_image(image, predictor, face_detector, model)
    #     print('prepare the output')
    #     # draw the facial landmarks
    #     landmarks_normalized = landmarks_normalized.astype(int) # landmarks after data normalization
    #     for (x, y) in landmarks_normalized:
    #         cv2.circle(img_normalized, (x, y), 5, (0, 255, 0), -1)
    #     face_patch_gaze,x,y = draw_gaze(img_normalized, pred_gaze_np)  # draw gaze direction on the normalized face image
    #     output_path = args.output
    #     print('save output image to: ', o)
    #     cv2.imwrite(o, face_patch_gaze)
    cap = cv.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    i=0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if not i%4:
<<<<<<< HEAD
            cam_file_name = 'camera_pocha.xml' if args.cam_id==0 else 'camera.xml'
            img_normalized, landmarks_normalized, pred_gaze_np =  pipeline_single_image(frame, predictor, face_detector, model, cam_file_name)
=======
            img_normalized, landmarks_normalized, pred_gaze_np =  pipeline_single_image(frame, predictor, face_detector, model)
>>>>>>> 82cc0bf5ec6fa51b77db93225daa572b70fb77ee
            _,x,y = draw_gaze(img_normalized, pred_gaze_np)

            x = max(min(x,max_x),min_x)
            y = max(min(y,max_y),min_y)

            norm_x = 224*(x - min_x)/(max_x-min_x)
            norm_y = 224*(y - min_y)/(max_y-min_y)   
            
            cv.circle(img_normalized, (int(norm_x), int(norm_y)), 5, (0, 255, 0), -1)
            gray = cv.cvtColor(img_normalized,cv.COLOR_RGB2RGBA)
            # Display the resulting frame
            cv.imshow('frame', cv.flip(gray,1))
            if cv.waitKey(1) == ord('q'):
                break
        else:
            cv.imshow('frame', cv.flip(gray,1))
            if cv.waitKey(1) == ord('q'):
                break
        i+=1
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
