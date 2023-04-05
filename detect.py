# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
from operator import index
import os
import sys
from pathlib import Path

import cv2
from cv2 import line
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd

from pytesseract import *
import imutils

isrotate = True
configs = '-l eng_ --psm 11 --oem 1 '

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
#--weights ../Weight/First/best.pt --img 4967 --conf 0.3 --source ../../../../../../../DSME_PID/p25.jpg

@torch.no_grad()

class Preprocess:
    srcimg = np.zeros((400,400,3), dtype=np.int)

    def __init__(self):
        self.result = np.zeros((400,400,3), dtype=np.int)

    def rotatefind(src_):
        src = src_.copy()
        if(isrotate):
            for i in range(0, 4):
                if(i==0):
                    results0 = image_to_data(src,output_type=Output.DICT, config=configs) #config='--psm 11 --oem 3' 
                if(i==1):
                    src = imutils.rotate_bound(src, 90)
                    results1 = image_to_data(src,output_type=Output.DICT, config=configs) #config='--psm 11 --oem 3' 
                    
                if(i==3):
                    src = imutils.rotate_bound(src, 270)
                    results2 = image_to_data(src,output_type=Output.DICT, config=configs) #config='--psm 11 --oem 3' 
                    
            return results0, results1, results2

    def find(src):
        results = image_to_data(src, output_type=Output.DICT, config=configs) #config='--psm 11 --oem 3' 
        return results

    def DrawResult(_results, _src, rot=0):
        if(rot==1):
            __src = imutils.rotate_bound(np.copy(_src), 90)
            
        elif(rot==3):
            __src = imutils.rotate_bound(np.copy(_src), 270)
        else:
            __src = np.copy(_src)

        for i in range(0, len(_results["text"])):
            x = _results["left"][i]
            y = _results["top"][i]
            w = _results["width"][i]
            h = _results["height"][i]
            text = _results["text"][i]
            tmpconf = float(_results["conf"][i])
            conf = int(tmpconf)
            if conf > 80.:
                # text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                cv2.rectangle(__src, (x, y), (x + w, y + h), (0, 255, 0), 2) 
                cv2.putText(__src, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,.5, (0, 0, 255),2)
                
        if(rot==1):
            __src = imutils.rotate_bound(__src, 270)
        elif(rot==3):
            __src = imutils.rotate_bound(__src, 90)
        else:
            pass
        return __src

    def DeleteResult(_results, _src, rot=0):
        if(rot==1):
            __src = imutils.rotate_bound(np.copy(_src), 90)
        
        elif(rot==3):
            __src = imutils.rotate_bound(np.copy(_src), 270)
        else:
            __src = np.copy(_src)

        for i in range(0, len(_results["text"])):
            x = _results["left"][i]
            y = _results["top"][i]
            w = _results["width"][i]
            h = _results["height"][i]
            text = _results["text"][i]
            tmpconf = float(_results["conf"][i])
            conf = int(tmpconf)
            if conf > 50.:
                # text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                cv2.rectangle(__src, (x, y), (x + w, y + h), (255, 255, 255),thickness=cv2.FILLED) 
                # cv2.putText(__src, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,.8, (0, 0, 255),2)
                
        if(rot==1):
            __src = imutils.rotate_bound(__src, 270)
        elif(rot==3):
            __src = imutils.rotate_bound(__src, 90)
        else:
            pass
        return __src
    
    def DrawLines(src):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 5000, 1500, apertureSize = 5, L2gradient = True)
        lines = cv2.HoughLinesP(canny, .01, np.pi / 180, 250, minLineLength = 10, maxLineGap = 100)
        for i in lines:
            cv2.line(src, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 90, 200), 2)
        return src
        
    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            pass
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y


        
def getContactpointList(box, line):
    
    xy1 = (int(box[2][0]),int(box[2][1]))
    xy2 = (int(box[3][0]),int(box[3][1]))
    xy3 = (line[0][0],line[0][1])
    xy4 = (line[0][2],line[0][3])
   
    iscontact = False
    xlinelength = np.abs(xy3[0]-xy4[0])
    ylinelength = np.abs(xy3[1]-xy4[1])
    contactxlist = []
    contactylist = []
    boxxlist = range(xy1[0],xy2[0])
    boxylist = range(xy1[1],xy2[1])
    largeX = xy3[0] if xy3[0]>xy4[0] else xy4[0]
    smallX = xy3[0] if xy3[0]<xy4[0] else xy4[0]
    largeY = xy3[1] if xy3[1]>xy4[1] else xy4[1]
    smallY = xy3[1] if xy3[1]<xy4[1] else xy4[1]
    linexlist = np.arange(smallX, largeX)
    lineylist = np.arange(smallY, largeY)
    if(xlinelength>50): # x3-x4가 50이상=x방향 길이가 김=가로
        if (xy1[1]<xy3[1]<xy2[1] or xy1[1]>xy3[1]>xy2[1]): #만약 선의 y값이 박스 사이에 들어있다면,
            for x in boxxlist:
                try:
                    tmp = np.where(linexlist==x)
                    if(np.size(tmp)!=0):
                        contactxlist.append(linexlist[tmp])
                except:
                    pass
        
    if(ylinelength>50): # y3-y4가 50이상=y방향 길이가 김=세로
        if (xy1[0]<xy3[0]<xy2[0] or xy1[0]>xy3[0]>xy2[0]): #만약 선의 x값이 박스 사이에 들어있다면,
            for y in boxylist:
                try:
                    tmp = np.where(lineylist==y)
                    if(np.size(tmp)!=0):
                        contactylist.append(lineylist[tmp])
                except:
                    pass
                
    return contactxlist, contactylist

def isContact(box, line):
    xy1 = (int(box[2][0]),int(box[2][1]))
    xy2 = (int(box[3][0]),int(box[3][1]))
    xy3 = (line[0][0],line[0][1])
    xy4 = (line[0][2],line[0][3])
   
    iscontact = False
    xlinelength = np.abs(xy3[0]-xy4[0])
    ylinelength = np.abs(xy3[1]-xy4[1])
    boxxlist = range(xy1[0],xy2[0])
    boxylist = range(xy1[1],xy2[1])
    largeX = xy3[0] if xy3[0]>xy4[0] else xy4[0]
    smallX = xy3[0] if xy3[0]<xy4[0] else xy4[0]
    largeY = xy3[1] if xy3[1]>xy4[1] else xy4[1]
    smallY = xy3[1] if xy3[1]<xy4[1] else xy4[1]
    linexlist = np.arange(smallX, largeX)
    lineylist = np.arange(smallY, largeY)
    if(xlinelength>50): # x3-x4가 50이상=x방향 길이가 김=가로
        if (xy1[1]<xy3[1]<xy2[1] or xy1[1]>xy3[1]>xy2[1]): #만약 선의 y값이 박스 사이에 들어있다면,
            for x in boxxlist:
                try:
                    tmp = np.where(linexlist==x)
                    if(np.size(tmp)!=0):
                        iscontact = True
                except:
                    pass
        
    if(ylinelength>50): # y3-y4가 50이상=y방향 길이가 김=세로
        if (xy1[0]<xy3[0]<xy2[0] or xy1[0]>xy3[0]>xy2[0]): #만약 선의 x값이 박스 사이에 들어있다면,
            for y in boxylist:
                try:
                    tmp = np.where(lineylist==y)
                    if(np.size(tmp)!=0):
                        iscontact = True
                except:
                    pass
                
    return iscontact

def getContactline(linelist1, linelist2):
    connectionresult = np.zeros((len(linelist1), len(linelist2)))
    
    for i in range(int(np.size(linelist1)/4)) :
        print("i :" + str(i))
        for j in range(int(np.size(linelist2)/4)):
            # xy1 = (linelist1[i][0][0],linelist1[i][0][1])
            # xy2 = (line1[0][2],line1[0][3])
            # xy3 = (line2[0][0],line2[0][1])
            # xy4 = (line2[0][2],line2[0][3])
            islinecontact = False
            linval13 = np.square(linelist1[i][0][0]-linelist2[j][0][0]) + np.square(linelist1[i][0][1]-linelist2[j][0][1])
            linval23 = np.square(linelist1[i][0][2]-linelist2[j][0][0]) + np.square(linelist1[i][0][3]-linelist2[j][0][1])
            linval14 = np.square(linelist1[i][0][0]-linelist2[j][0][2]) + np.square(linelist1[i][0][3]-linelist2[j][0][1])
            linval24 = np.square(linelist1[i][0][2]-linelist2[j][0][2]) + np.square(linelist1[i][0][3]-linelist2[j][0][3])
            if(linval13<6 or linval23<6 or linval14<6 or linval24<6):
                connectionresult[i][j] = 1
    
    return connectionresult
    
def getContactList(boxlist, linelist):
    connectionresult = np.zeros((len(boxlist), len(linelist)))
    
    for i in range(len(boxlist)) :#box in boxlist:
        print("i :" + str(i))
        for j in range(int(np.size(linelist)/4)):
            #print("j :" + str(j))
            iscontact = isContact(boxlist[i], linelist[j])
            if iscontact:
                #linelistResult = np.delete(linelist, j, 0)
                connectionresult[i][j] = 1
                print(str(i)+"th symbol have connection with line number "+str(j))
                
    return connectionresult


def ListModify(boxlist, linelist):
    linelistResult = linelist.copy()
    for i in range(len(boxlist)) :#box in boxlist:
        print("i :" + str(i))
        for j in range(int(np.size(linelist)/4)):
            #print("j :" + str(j))
            Xlist, Ylist = getContactpointList(boxlist[i], linelist[j])
            if (np.size(Ylist)>1 or np.size(Xlist)>1):
                if np.size(Xlist)==0 and np.size(Ylist)==0: 
                    V = False
                    H = False
                elif np.size(Ylist)==0 and np.size(Xlist)!=0: 
                    V = False
                    H = True
                else :
                    V = True
                    H = False
                
                if V:
                    #linelistResult = np.delete(linelist, j, 0)
                    tmpxy2 = linelist[j][0][0:2]
                    newxy = np.array([], dtype=np.int32)
                    #newxy.clear()
                    newxy = np.append(newxy, linelist[j][0][0])
                    newxy = np.append(newxy, Ylist[1])
                    linelistResult[j][0][3] = newxy[1]
                    newxy_ = np.array([], dtype=np.int32)
                    #newxy_.clear()
                    newxy_ = np.append(newxy_, linelist[j][0][0])
                    newxy_ = np.append(newxy_, Ylist[-1])
                    newxyxy = np.array([[[newxy_[0], newxy_[1], tmpxy2[0], tmpxy2[1]]]])
                    linelistResult = np.append(linelistResult, newxyxy, axis=0)
                    #print("have contact with vertical line")
                elif H:
                    #linelist = np.delete(linelist, j, 0)
                    tmpxy2 = linelist[j][0][2:4]
                    newxy = np.array([], dtype=np.int32)
                    #newxy.clear()
                    newxy = np.append(newxy, Xlist[1])
                    newxy = np.append(newxy, linelist[j][0][1])
                    linelistResult[j][0][2] = newxy[0]
                    linelistResult[j][0][3] = newxy[1]
                    newxy_ = np.array([], dtype=np.int32)
                    #newxy_.clear()
                    newxy_ = np.append(newxy_, Xlist[-1])
                    newxy_ = np.append(newxy_, linelist[j][0][1])
                    newxyxy = np.array([[[newxy_[0], newxy_[1], tmpxy2[0], tmpxy2[1]]]])
                    linelistResult = np.append(linelistResult, newxyxy, axis=0)
                    #print("have contact with horizontal line")
                
            
    return linelistResult
    
def RemoveDuplicate(lines):
    for line_ in range(int(np.size(lines)/4)):
        for line in range(int(np.size(lines)/4)):
            if line == 0 or line==line_:
                continue
            if(np.abs(lines[line][0][0]-lines[line][0][2])>50 and 
                np.abs(lines[line_][0][0]-lines[line_][0][2])>50 and
                np.abs(lines[line_][0][1]-lines[line][0][1])<5 and
                np.abs(lines[line_][0][3]-lines[line][0][3])<5 or 
                np.abs(lines[line][0][1]-lines[line][0][3])>50 and 
                np.abs(lines[line_][0][1]-lines[line_][0][3])>50 and
                np.abs(lines[line_][0][0]-lines[line][0][0])<5 and
                np.abs(lines[line_][0][2]-lines[line][0][2])<5):
                lines[line][0][0] = 0#x1
                lines[line][0][2] = 0#x2
                lines[line][0][1] = 0#y1
                lines[line][0][3] = 0#y2
    line = 0
    result = lines.copy()
    deletelist = np.array([], dtype=np.int32)
    while(True):
        try :
            if (lines[line][0][0] == 0 and lines[line][0][2] == 0 and lines[line][0][1] == 0 and lines[line][0][3] == 0):
                deletelist = np.append(deletelist, int(line))
                line += 1
            else : line += 1
        except : break
        
    result = np.delete(result, deletelist, axis=0)
                
    return result

def run(weights=ROOT / '../Weight/Third/best.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=4967,  # inference size (pixels)
        conf_thres=0.15,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()


    
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = 1  # batch_size
 
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            results0, results1, results2 = Preprocess.rotatefind(im0)
            im0 = Preprocess.DrawResult(results0, im0 , rot=0)
            im0 = Preprocess.DrawResult(results1, im0 , rot=1)      
            #src = Preprocess.DrawResult(results0, image)
            imk = imc.copy()#라인 인식용
            
                               
            
                              
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            annotator_imk = Annotator(imk, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                boxlist = {}
                num = 0
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    boxlist[num] = (int(cls), float(conf), (int(xyxy[0]),int(xyxy[1])), (int(xyxy[2]),int(xyxy[3])))
                    num +=1
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if c==18 or c==17:
                            pass
                        else:
                            annotator_imk.box_delete(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            
                
                imk = Preprocess.DeleteResult(results0, imk , rot=0)
                imk = Preprocess.DeleteResult(results1, imk , rot=1)
                
                # cv2.imshow("",imk)
                # cv2.waitKey()
                
                gray = cv2.cvtColor(imk, cv2.COLOR_BGR2GRAY)
                canny = cv2.Canny(gray, 5000, 1500, apertureSize = 5, L2gradient = True)
                # cv2.imshow("",canny)
                # cv2.waitKey()
                lines = cv2.HoughLinesP(canny, .01, np.pi / 180, 150, minLineLength = 20, maxLineGap = 40)
                # lines = RemoveDuplicate(lines)
                
                #lines = ListModify(boxlist, lines)
                #lines = ListModify(boxlist, lines)
                #lines = ListModify(boxlist, lines)
                #lines = ListModify(boxlist, lines)
                #lines = ListModify(boxlist, lines)
                
                mat1 = getContactline(lines, lines)
                mat2 = getContactList(boxlist, lines)
                print("lines connection")
                print(mat1)
                print("")
                
                print("Symbol, lines connection")
                print(mat2)
                print("")
                for i in lines:
                    random_color=list(np.random.choice(range(255),size=4))
                    cv2.line(im0, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (int(random_color[0]),int(random_color[1]),int(random_color[2])), 2) #(0, 90, 200)
                # cv2.imshow("",imk)
                # cv2.waitKey()
                
                
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            linescont = pd.DataFrame(mat1)
            boxlinecont = pd.DataFrame(mat2)
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    linescont.to_excel(save_path+'linescontact.xlsx', index=False)
                    boxlinecont.to_excel(save_path+'boxlinescontact.xlsx', index=False)
                    cv2.imwrite(save_path, im0)
                

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / '../Weight/Third/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / '../../../../../../../DSME_PID', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[4967], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.55, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
