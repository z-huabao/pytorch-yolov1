#python3
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def voc_ap(recall, precision, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0.,1.1,0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall>=t])
            ap = ap + p/11.
    else:
        # correct ap caculation
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        for i in range(mpre.size -1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def voc_eval(targets, predicts, classes, threshold=0.5, use_07_metric=False):
    """
    get mAP: https://www.zhihu.com/question/53405779
    targets: [[file,x1,y1,x2,y2,cls], ...]
    predicts: [[file,x1,y1,x2,y2,cls, prob], ...]
    """

    def parse_target(targets):
        """out: {0: {imgfile: [[x1,y1,x2,y2], ...], ...}, ...}"""
        out = {}
        for ann in targets:
            f, c, bbox = ann[0], int(ann[5]), ann[1:5]
            out.setdefault(int(c), {})
            out[c].setdefault(f, [])
            out[c][f].append(bbox)
        return out

    def parse_predict(predicts):
        """out: {0:[[imgfile,x1,y1,x2,y2,cls,prob],...], 1:[[],...]}"""
        out = {}
        for ann in predicts:
            f, c = ann[0], int(ann[5])
            out.setdefault(c, [])
            out[c].append(ann)
        return out

    targets = parse_target(targets)
    predicts = parse_predict(predicts)

    aps = []
    for cls, cls_name in enumerate(classes):
        targets_cls = targets.get(cls)  # {file1: [box1,box2,...], file2:...}
        num_targets = float(sum(len(b) for b in targets_cls.values()))

        pred_cls = predicts.get(cls)
        # 如果这个类别一个都没有检测到的异常情况
        if not pred_cls or not targets_cls or len(pred_cls) == 0:
            ap = -1
            aps += [ap]
            print('---class {} ap {}---'.format(cls_name,ap))
            continue

        # probs从大到小排序，对应到tp
        order = np.argsort([r[-1] for r in pred_cls])[::-1]

        tp = []  # true predict
        for bbox in (pred_cls[i][:5] for i in order):  # bbox: [file,x1,y1,x2,y2]
            tp.append(0)

            imgfile = bbox[0]
            pdbox = np.int32(bbox[1:])

            gtboxes = targets_cls.get(imgfile) or []  # gt: ground truth
            for gtbox in gtboxes:
                # compute overlaps between to box(IoU)
                ixmin = np.maximum(gtbox[0], pdbox[0])
                iymin = np.maximum(gtbox[1], pdbox[1])
                ixmax = np.minimum(gtbox[2], pdbox[2])
                iymax = np.minimum(gtbox[3], pdbox[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                union = (pdbox[2]-pdbox[0]+1.)*(pdbox[3]-pdbox[1]+1.) + (gtbox[2]-gtbox[0]+1.)*(gtbox[3]-gtbox[1]+1.) - inters
                if union == 0:
                    print('error:', pdbox, gtbox)

                overlaps = inters/union
                if overlaps > threshold:
                    tp[-1] = 1
                    gtboxes.remove(gtbox) #这个框已经匹配到了，不能再匹配
                    break

        fp = 1 - np.array(tp)  # false predict
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recall = tp / num_targets
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        # print(recall[-5:], precision[-5:])
        ap = voc_ap(recall, precision, use_07_metric)
        aps += [ap]
        print('---class {} ap {}---'.format(cls_name, ap))

    print('---map {}---\n'.format(np.mean(aps)))

def load_labels(*args, dtype='yolo'):
    def load_yolo_labels(txt):
        """load bbox and labels from *.txt annotation file"""
        with open(txt, 'r') as f:
            data = [l.strip().split() for l in f.readlines()]

        fs, anns = [], []
        for ann in data:  # [imgfile, x1, y1, x2, y2, cls, x1, y1, x2, ...]
            if ann:
                imgfile = ann[0]
                fs.append(imgfile)
                for box in np.int32(ann[1:]).reshape(-1, 5):
                    cls = box[-1]  # box: (x1, y1, x2, y2, cls)
                    anns.append([imgfile] + box.tolist())

        from data.classes import VOC_CLASSES
        return fs, anns, VOC_CLASSES

    def load_coco_labels(file_):
        data = json.load(open(file_, 'r'))
        id2file = {d['id']: d['file_name'] for d in data['images']}
        anns = []
        for ann in data['annotations']:
            f = id2file[ann['image_id']]
            b = np.int32(ann['bbox'])
            b[2:] += b[:2]
            c = ann['category_id'] - 1
            anns.append([f] + b.tolist() + [c])

        classes = {c['id']-1: c['name'] for c in data['catogories']}
        return list(id2file.value()), anns, [classes[i] for i in range(len(classes))]

    if dtype == 'yolo':
        return load_yolo_labels(*args)
    elif dtype == 'coco':
        return load_coco_labels(*args)


if __name__ == '__main__':
    from predict import *

    print('---prepare targets---')
    image_list, targets, classes = load_labels('data/voc2007test.txt', dtype='yolo')

    print('---prepare predicts---')
    predicts_json = 'data/predicts.json'
    if not os.path.exists(predicts_json) or 0:
        # run network, get results and save in data/predicts.json
        print('load model...')
        backbone = Backbone('resnet50', 'layer4', pretrained=False)
        net = YoloV1(backbone)
        net.load_state_dict(torch.load('./output/best.pth', map_location='cpu'))
        net.eval()
        if USE_GPU:
            print('use gpu')
            net.cuda()

        print('---start test---')
        predicts = []
        for imgfile in tqdm(image_list[:]):
            image = cv2.imread('data/images/' + imgfile)
            result = predict_gpu(net, image, 0.1)

            order = np.argsort([r[-1] for r in result])[::-1]

            for r in (result[i] for i in order):  # r: x1,y1,x2,y3,cls,prob
                predicts.append([imgfile] + [str(i) for i in r])

        with open(predicts_json, 'w') as f:
            json.dump(predicts, f, indent=2)
            print('saved predicts')

    print('load predicts from: ', predicts_json)
    with open(predicts_json, 'r') as f:
        predicts = json.load(f)

    print('---start evaluate---')
    voc_eval(targets, predicts, classes)

