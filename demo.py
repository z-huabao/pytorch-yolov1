#encoding:utf-8
#python3
import time
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from torch.autograd import Variable

from nets import Backbone, YoloV1
from data.classes import VOC_CLASSES, COLOR

USE_GPU = torch.cuda.is_available() and 0


def decoder(pred):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    grid_num = 14
    cell_size = 1. / grid_num
    pred = pred.data.squeeze(0) #7x7x30

    contain = pred[:,:,[4,9]]  # two box every grid cell
    mask1 = contain > 0.1 #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框

    boxes, cls_indexs, probs = [], [], []
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):  # two box every grid cell
                if mask[i,j,b] == 1:
                    #print(i,j,b)
                    box = pred[i, j, b*5:b*5+4]
                    box[:2] = (torch.FloatTensor([j, i]) + box[:2]) * cell_size

                    # convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box[:2] -= 0.5*box[2:]
                    box[2:] += box[:2]

                    max_prob, cls_index = torch.max(pred[i, j, 10:],0)
                    prob = pred[i, j, b*5+4] * max_prob
                    if prob > 0.1:
                        boxes.append(box.view(1,4))
                        cls_indexs.append(cls_index)
                        probs.append(prob)
    if len(boxes):
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.tensor(probs)
        cls_indexs = torch.tensor(cls_indexs)
    else:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    keep = nms(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]

def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0]
            keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

def predict_gpu(model, image):
    """predict one image"""
    h, w, _ = image.shape

    img = cv2.resize(image, (448,448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123,117,104)  # RGB
    img = img - np.array(mean, dtype=np.float32)
    img = img.transpose((2, 0, 1))  # [3, 448, 448]
    img = torch.from_numpy(img[None, ...])  # [1, 3, 448, 448]
    if USE_GPU:
        img = img.cuda()

    pred = model(img) # [1, 7, 7, 30]
    # print(pred.shape)
    boxes, cls_indexs, probs = decoder(pred.cpu())

    result = []
    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1), (x2,y2), VOC_CLASSES[cls_index], prob])
    return result

def draw_image(image, result):
    for left_up, right_bottom, class_name, prob in result:
        color = COLOR[VOC_CLASSES.index(class_name)]
        label = class_name+str(round(prob,2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1])

        cv2.rectangle(image, left_up, right_bottom, color, 2)

        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imshow('Press `s` to save; `q` to quit', image)

    key = cv2.waitKey(10)
    if key == ord('s'):
        cv2.imwrite('result.jpg', image)
    elif key == ord('q'):
        quit(0)

if __name__ == '__main__':
    cam = cv2.VideoCapture(-1)
    image = cam.read()[1]
    assert image is not None, str(cam)
    print('camera resolution:', image.shape)

    backbone = Backbone('resnet50', 'layer4', pretrained=True)
    net = YoloV1(backbone)

    print('loading model...')
    # net.load_state_dict(torch.load('./output/%s-best.pth' % net.name, map_location='cpu'))
    net.load_state_dict(torch.load('./output/best.pth', map_location='cpu'))
    net.eval()

    import torchsummary, torchviz
    torchsummary.summary(net, (3,448,448), device='cpu')
    torchviz.make_dot(net, torch.zeros([1,3,448,448])).view()

    if USE_GPU:
        net.cuda()

    while True:
        image = cam.read()[1]
        assert image is not None, str(cam)

        t1 = time.time()
        result = predict_gpu(net, image)
        t2 = time.time()

        print('detect time:', t2 - t1)
        draw_image(image, result)
