#python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchviz
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchsummary import summary
from torch.autograd import Variable
from tqdm import tqdm

from visualize import Visualizer

from nets import Backbone, YoloV1
from yoloLoss import yoloLoss
from data.dataset import yoloDataset

USE_GPU = torch.cuda.is_available()

file_root = './data/images/'
learning_rate = 0.1
num_epochs = 50
batch_size = 22


# build yolo-v1 network
backbone = Backbone('resnet50', 'layer4', pretrained=True)
net = YoloV1(backbone)
net.train()
criterion = yoloLoss(7,2,5,0.5)

# net.load_state_dict(torch.load('output/%s-best.pth' % net.name))
# print('load pre-trined model')
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())
summary(net, (3, 448, 448), device='cpu')

if USE_GPU:
    net.cuda()

# view network
# torchviz.make_dot(net, torch.rand(1,3,448,448).cuda()).view()
# quit(0)

# initial optimizer
params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

# build dataset generator
train_dataset = yoloDataset(root=file_root,
        list_file=['data/voc2012.txt','data/voc2007.txt'], is_train=True, transform=[])
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)

test_dataset = yoloDataset(root=file_root,
        list_file='data/voc2007test.txt', is_train=False, transform=[])
test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers=4)

print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))

# log, save validate loss every epoch
logfile = open('output/log.txt', 'w')

num_iter = 0
print_every_step = 5
vis = Visualizer(env='yolo-v1')
best_test_loss = np.inf

# train by epochs
for epoch in range(num_epochs):
    net.train()
    # learning rate schedule
    if epoch == 30:
        learning_rate=0.01
    if epoch == 40:
        learning_rate=0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    total_loss = 0.
    # train by iter on every epoch
    for i, (images, target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if USE_GPU:
            images,target = images.cuda(),target.cuda()

        # forward
        pred = net(images)  # [1,7,7,30]
        loss = criterion(pred,target)
        total_loss += loss.item()
        vis.plot_train_val(loss_train=loss.item())  # train plot

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logout
        if (i+1) % print_every_step == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'%(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
            num_iter += 1

    # save checkpoint model
    torch.save(net.state_dict(), './output/%s.pth' % net.name)

    # validate
    validation_loss = 0.0
    net.eval()
    for i, (images, target) in tqdm(enumerate(test_loader)):
        images = Variable(images,volatile=True)
        target = Variable(target,volatile=True)
        if USE_GPU:
            images, target = images.cuda(), target.cuda()

        pred = net(images)
        loss = criterion(pred,target)
        validation_loss += loss.item()
    validation_loss /= len(test_loader)
    vis.plot_train_val(loss_val=validation_loss)  # val plot

    # save best checkpoint model
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(), './output/%s-best.pth' % net.name)

    # log val loss
    logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
    logfile.flush()

logfile.close()

