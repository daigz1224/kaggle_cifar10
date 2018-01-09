import os
import math
import mxnet as mx
from mxnet import image
from mxnet import nd, gluon, autograd, init
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.gluon import nn
from tensorboardX import SummaryWriter
import numpy as np
import shutil

def transform_train(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, 
                        rand_crop=True, rand_resize=True, rand_mirror=True,
                        mean=np.array([0.4914, 0.4822, 0.4465]), 
                        std=np.array([0.2023, 0.1994, 0.2010]), 
                        brightness=0, contrast=0, 
                        saturation=0, hue=0, 
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

# 测试时，无需对图像做标准化以外的增强数据处理。
def transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), 
                        mean=np.array([0.4914, 0.4822, 0.4465]), 
                        std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

## 使用Gluon读取整理后的数据集
data_dir = '../data'
input_dir = 'train_valid_test'

batch_size = 128
input_str = data_dir + '/' + input_dir + '/'

# 读取原始图像文件。flag=1 说明输入图像有三个通道（彩色）。
train_ds = ImageFolderDataset(input_str + 'train', flag=1, transform=transform_train)
valid_ds = ImageFolderDataset(input_str + 'valid', flag=1, transform=transform_test)
train_valid_ds = ImageFolderDataset(input_str + 'train_valid', flag=1, transform=transform_train)
test_ds = ImageFolderDataset(input_str + 'test', flag=1, transform=transform_test)

train_data = DataLoader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = DataLoader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = DataLoader(train_valid_ds, batch_size, shuffle=True, last_batch='keep')
test_data = DataLoader(test_ds, batch_size, shuffle=False, last_batch='keep')

criterion = gluon.loss.SoftmaxCrossEntropyLoss()

## 导入模型
from model_resnet import ResNet164_v2

model = ResNet164_v2(10)
model.initialize(ctx=mx.gpu(1), init=mx.initializer.Xavier())
model.hybridize()

## 训练函数
import datetime

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            acc += nd.sum(net(X).argmax(axis=1)==y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read() # don't push too many operators into backend
    return acc.asscalar() / n

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})

    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = criterion(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_acc = evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data), valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))

## 训练调参
# ctx = mx.gpu()
# num_epochs = 1
# learning_rate = 0.1
# weight_decay = 5e-4
# lr_period = 80
# lr_decay = 0.1
#net = get_net(ctx, ResNet164_v2)
#net.hybridize()
#train(net, train_data, valid_data, num_epochs, learning_rate, 
#      weight_decay, ctx, lr_period, lr_decay)

## 全集训练并对测试集分类
import pandas as pd

ctx = mx.gpu()
num_epochs = 300
learning_rate = 0.1
weight_decay = 1e-4
lr_period = 120
lr_decay = 0.1
net = get_net(ctx, ResNet164_v2)
net.hybridize()
train(net, train_valid_data, None, num_epochs, learning_rate, 
      weight_decay, ctx, lr_period, lr_decay)

preds = []
for data, label in test_data:
    output = net(data.as_in_context(ctx))
    preds.extend(output.argmax(axis=1).astype(int).asnumpy())

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x:str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)