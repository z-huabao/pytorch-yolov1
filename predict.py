#encoding:utf-8
#python3
import time
import cv2
import torch
import numpy as np
from nets import Backbone, YoloV1
from data.classes import VOC_CLASSES, COLOR

USE_GPU = torch.cuda.is_available() and 1

def nms(bboxes,scores,threshold=0.5):
    '''
    滤掉重叠的框
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

def decoder(pred, thresh):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    grid_num = 14
    cell_size = 1. / grid_num
    pred = pred.data.squeeze(0) #7x7x30

    contain = pred[:,:,[4,9]]  # two box every grid cell
    mask1 = contain > thresh #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)

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
                    if prob > thresh:
                        boxes.append(box.view(1,4))
                        cls_indexs.append(cls_index)
                        probs.append(prob)

    # convert boxes and probs to torch array
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

def predict_gpu(model, image, thresh=0.1):
    """predict one image"""
    h, w, _ = image.shape

    # preprocess image
    img = cv2.resize(image, (448,448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123,117,104)  # RGB
    img = img - np.array(mean, dtype=np.float32)
    img = img.transpose((2, 0, 1))  # [3, 448, 448]
    img = torch.from_numpy(img[None, ...])  # [1, 3, 448, 448]
    if USE_GPU:
        img = img.cuda()

    # run network
    pred = model(img) # [1, 14, 14, 30]
    # print(pred.shape)
    boxes, cls_indexs, probs = decoder(pred.cpu(), thresh)

    # manage output
    boxes[boxes < 0] = 0
    boxes[boxes > 1] = 1
    boxes = np.int32(np.float32(boxes) * [w,h,w,h])
    result = [(x1, y1, x2, y2, int(cls_index), float(prob)) for
                (x1, y1, x2, y2), cls_index, prob in zip(boxes, cls_indexs, probs)]
    return result

def draw_image(image, result, is_show=True):
    for x1, y1, x2, y2, cls, prob in result:
        color = COLOR[cls]
        label = VOC_CLASSES[cls]+str(round(prob,2))

        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (x1, y1- text_size[1])

        # draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # draw label
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    if is_show:
        cv2.imshow('Press `s` to save; `q` to quit', image)

        key = cv2.waitKey(10)
        if key == ord('s'):
            cv2.imwrite('demo.jpg', image)
        elif key == ord('q'):
            quit(0)
    else:
        return image

if __name__ == '__main__':
    print('load model...')
    backbone = Backbone('resnet50', 'layer4', pretrained=False)
    net = YoloV1(backbone)
    net.load_state_dict(torch.load('./output/best.pth', map_location='cpu'))
    net.eval()
    if USE_GPU:
        print('use gpu')
        net.cuda()

    # load image
    image_name = 'data/dog.jpg'
    image = cv2.imread(image_name)

    print('predicting...')
    result = predict_gpu(net, image)

    image = draw_image(image, result, False)
    cv2.imwrite('dog.jpg',image)

