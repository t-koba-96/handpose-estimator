from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter

args = {'dataset':'BCCD',  # VOC → BCCD
        'basenet':'vgg16_reducedfc.pth',
        'batch_size':12,
        'resume':'',
        'start_iter':0,
        'num_workers':0,  # 4 → 0
        'cuda':True,  # Macの場合False
        'lr':5e-4,
        'momentum':0.9,
        'weight_decay':5e-4,
        'gamma':0.1,
        'save_folder':'weights/'
       }

if torch.cuda.is_available():
    if args['cuda']:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args['cuda']:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args['save_folder']):
    os.mkdir(args['save_folder'])

# 訓練データの読み込み
cfg = bccd
dataset = VOCDetection(root=VOC_ROOT,
                       transform=SSDAugmentation(cfg['min_dim'],
                                                 MEANS))

# ネットワークの定義
ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
net = ssd_net
print(net)

if args['cuda']:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True
    
# パラメータのロード
if args['resume']:
    print('Resuming training, loading {}...'.format(args['resume']))
    ssd_net.load_weights(args['resume'])
else:
    vgg_weights = torch.load(args['save_folder'] + args['basenet'])
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)
    
if args['cuda']:
    net = net.cuda()
    
def adjust_learning_rate(optimizer, gamma, step):
    lr = args['lr'] * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
        
if not args['resume']:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)
    
# 最適化パラメータの設定
optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=args['momentum'],
                      weight_decay=args['weight_decay'])

# 損失関数の設定
criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                         False, args['cuda'])

net.train()
# loss counters
loc_loss = 0
conf_loss = 0
epoch = 0
print('Loading the dataset...')

epoch_size = len(dataset) // args['batch_size']
print('Training SSD on:', dataset.name)
print('Using the specified args:')
print(args)

step_index = 0

# 訓練データの読み込み
data_loader = data.DataLoader(dataset, args['batch_size'],
                              num_workers=args['num_workers'],
                              shuffle=True, collate_fn=detection_collate,
                              pin_memory=True)

#tensorboard
writer = SummaryWriter(os.path.join("logs",args['dataset']))

# 学習の開始
batch_iterator = None
# iterationでループして、cfg['max_iter']まで学習する
for iteration in range(args['start_iter'], cfg['max_iter']):
    # 学習開始時または1epoch終了後にdata_loaderから訓練データをロードする
    if (not batch_iterator) or (iteration % epoch_size ==0):
      batch_iterator = iter(data_loader)
      loc_loss = 0
      conf_loss = 0
      epoch += 1

    if iteration in cfg['lr_steps']:
        step_index += 1
        adjust_learning_rate(optimizer, args['gamma'], step_index)

    # load train data
    # バッチサイズ分の訓練データをload
    images, targets = next(batch_iterator)

    if args['cuda']:
        images = Variable(images.cuda())
        targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
    else:
        images = Variable(images)
        targets = [Variable(ann, volatile=True) for ann in targets]
    # forward
    t0 = time.time()
    out = net(images)
    # backprop
    optimizer.zero_grad()
    loss_l, loss_c = criterion(out, targets)
    loss = loss_l + loss_c
    loss.backward()
    optimizer.step()
    t1 = time.time()
    loc_loss += loss_l.item()
    conf_loss += loss_c.item()


    #ログの出力
    if iteration % 10 == 0:
        print('timer: %.4f sec.' % (t1 - t0))
        print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')
        writer.add_scalar('loss/train_loss', loss.item() , iteration)

print('Finished Training')
writer.close()

# 学習済みモデルの保存
torch.save(ssd_net.state_dict(),
           args['save_folder'] + '' + args['dataset'] + '.pth')