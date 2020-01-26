import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
 
from python import util
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import os
import glob
import copy
from ssd import ssd_setting
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
from argparse import ArgumentParser
 

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="voc")
    return argparser.parse_args()

 
# 関数 detect    
def detect(net, image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = cv2.resize(image, (300, 300)).astype(np.float32)  # 300*300にリサイズ
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)  # [300,300,3]→[3,300,300]
    xx = Variable(x.unsqueeze(0))     # [3,300,300]→[1,3,300,300]    
    # 順伝播を実行し、推論結果を出力    
    y = net(xx)
    # 推論結果をdetectionsに格納
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

    return detections , scale
      

def draw_ssd_bbox(image, detections, scale, labels):

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # バウンディングボックスとクラス名を表示
    canvas = copy.deepcopy(image)
    for i in range(detections.size(1)):
        j = 0
        # 確信度confが0.6以上のボックスを表示
        # jは確信度上位200件のボックスのインデックス
        # detections[0,i,j]は[conf,xmin,ymin,xmax,ymax]の形状
        while detections[0,i,j,0] >= 0.6:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            color = colors[i]
            f_h, f_w, f_c = canvas.shape
            cv2.rectangle(canvas, (pt[0], pt[1]), (pt[2], pt[3]), (color[0]*255,color[1]*255,color[2]*255),2)
            fig = Figure(figsize=plt.figaspect(canvas))
            fig.subplots_adjust(0, 0, 1, 1)
            fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
            bg = FigureCanvas(fig)
            ax = fig.subplots()
            ax.axis('off')
            ax.imshow(canvas)
            width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()
            ax.text(pt[0], pt[1] - 5, label_name, size = 15, color = color)
            bg.draw()
            canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            canvas = cv2.resize(canvas,(f_w, f_h))
            j = j + 1

    return canvas

 
def main():

    args = get_option()
        
    #モデル読み込み
    net = ssd_setting.build_ssd('test', 300, 21) 

    if args.dataset == "voc":
        net.load_weights('./weights/ssd/pretrained.pth')
        from ssd.data import VOC_CLASSES as labels
    else:
        print("chose available pretrained model")
        return    

    files = sorted(glob.glob('./demo/input/*.jpg'))
    count = 1
    for i, file in enumerate (files):
        img_name = os.path.split(file)[1]
        image = cv2.imread(file, cv2.IMREAD_COLOR)          
        detections , scale = detect(net, image)
        output_img = draw_ssd_bbox(image, detections, scale, labels)
        cv2.imwrite(os.path.join('demo','output',img_name),output_img)
        print(count)
        count +=1
    
    return
 
if __name__ == '__main__':
    main()  