import sys
sys.path.insert(0, 'python')
import cv2
import os
import model
import util
import glob
from hand import Hand
from body import Body
import matplotlib.pyplot as plt
import copy
import numpy as np
import detect_object
from ssd import ssd_setting
from mmdet.apis import init_detector, inference_detector
from argparse import ArgumentParser

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument("mode", type=str, help="handpose or openpose")
    argparser.add_argument("--config_file", type=str, help="specify the config_file", default="./src/configs/ttfnet.py")
    argparser.add_argument("--weight_file", type=str, help="specify the checkpoint_file", default="./weights/ttfnet/latest.pth")

    #rpn 
    argparser.add_argument("--object_detection", type=str, help="true or not", default=True)
    argparser.add_argument("--ssd_weight_path", type=str, help="specify the weight_file", default='./weights/ssd/pretrained.pth')


    #openpose weight
    argparser.add_argument("--body_weight_path", type=str, help="specify the weight_file", default="./weights/openpose/body_pose_model.pth")
    argparser.add_argument("--hand_weight_path", type=str, help="specify the weight_file", default="./weights/openpose/hand_pose_model.pth")
   
    return argparser.parse_args()


def detect(oriImg, image_name, mode, model, hand_estimation, body_estimation, rpn, labels):
    

    height, width, channel = oriImg.shape
    canvas = copy.deepcopy(oriImg)

    if mode == "openpose":
        candidate, subset = body_estimation(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)
        # detect hand
        hands_list = util.handDetect(candidate, subset, oriImg)
        # detect object
        detections , scale = detect_object.detect(rpn, oriImg)

        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
           
            peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            
            all_hand_peaks.append(peaks)


    elif mode == "handpose":
        bounding_box = inference_detector(model, oriImg)

        # detect object
        detections , scale = detect_object.detect(rpn, oriImg)


        all_hand_peaks = []
        for xmin, ymin, xmax, ymax, prob in bounding_box[0]:
            if prob < 0.4:
                continue
            
            fixed_xmin = int(xmin) - 50
            if fixed_xmin <=0:
                fixed_xmin = 1
            fixed_xmax = int(xmax) + 50
            if fixed_xmax >= width:
                fixed_xmax = width - 1
            fixed_ymin = int(ymin) - 50
            if fixed_ymin <=0:
                fixed_ymin = 1
            fixed_ymax = int(ymax) + 50
            if fixed_ymax >= height:
                fixed_ymax = height - 1


            cv2.rectangle(canvas, (fixed_xmin, fixed_ymin), (fixed_xmax, fixed_ymax), (0, 0, 255), 2)
            peaks = hand_estimation(oriImg[fixed_ymin:fixed_ymax, fixed_xmin:fixed_xmax, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+fixed_xmin)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+fixed_ymin)

            all_hand_peaks.append(peaks)


    canvas = detect_object.draw_ssd_bbox(canvas, detections, scale, labels)
    if all_hand_peaks:
        canvas = util.draw_handpose(canvas, all_hand_peaks)
        canvas = cv2.resize(canvas,(width, height))
        
    cv2.imwrite(os.path.join('demo','output',image_name),canvas)
    print('done')
    return

def main():

    args = get_option()

    if args.object_detection:
        rpn = ssd_setting.build_ssd('test', 300, 21) 
        rpn.load_weights(args.ssd_weight_path)
        from ssd.data import VOC_CLASSES as labels
    else:
        rpn = False


    #model load
    model = init_detector(args.config_file, args.weight_file, device="cuda:0")

    hand_estimation = Hand(args.hand_weight_path)
    if args.mode == "openpose":
        body_estimation = Body(args.body_weight_path)
    else:
        body_estimation = False
    
    files = sorted(glob.glob('./demo/input/*.jpg'))
    print('hello')
    count = 1
    for i, file in enumerate (files):
        img_name = os.path.split(file)[1]
        image = cv2.imread(file, cv2.IMREAD_COLOR)          
        detect(image, img_name, args.mode, model, hand_estimation, body_estimation, rpn, labels)
        print(count)
        count +=1

if __name__ == "__main__":
    main()