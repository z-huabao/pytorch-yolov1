#encoding:utf-8
#python3
import torchsummary, torchviz
from predict import *

cam = cv2.VideoCapture(0)
image = cam.read()[1]
assert image is not None, str(cam)
print('camera resolution:', image.shape)

backbone = Backbone('resnet50', 'layer4', pretrained=True)
net = YoloV1(backbone)

print('loading model...')
#net.load_state_dict(torch.load('./output/%s-best.pth' % net.name, map_location='cpu'))
net.load_state_dict(torch.load('./output/best.pth', map_location='cpu'))
net.eval()

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
