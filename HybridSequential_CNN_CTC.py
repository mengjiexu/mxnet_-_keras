# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss
from mxnet.gluon import nn
import numpy as  np
import os
import sys
import gluonbook as gb


class Resnet1D(nn.HybridBlock):
    def __init__(self, num_channels, **kwargs):
        super(Resnet1D, self).__init__(**kwargs)
        self.conv1 = nn.Conv1D(num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1D(num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    # def forward(self, x):
    #     y = nd.relu(self.bn1(self.conv1(x)))
    #     y = self.bn2(self.conv2(y))
    #     return nd.relu(y + x)

    def hybrid_forward(self, F, x, *args, **kwargs):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + x)


class CBR(nn.HybridBlock):
    def __init__(self, num_channels, kernel_size=3, padding=1, **kwargs):
        super(CBR, self).__init__(**kwargs)
        self.conv = nn.Conv1D(num_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm()
        self.relu = nn.Activation('relu')

    # def forward(self, x):
    #     return self.relu(self.bn(self.conv(x)))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.relu(self.bn(self.conv(x)))


class SwapAxes(nn.HybridBlock):
    def __init__(self, dim1, dim2):
        super(SwapAxes, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    # def forward(self, x):
    #     return nd.swapaxes(x, self.dim1, self.dim2)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.swapaxes(x, self.dim1, self.dim2)


with mx.Context(mx.cpu(0)):
    model = nn.HybridSequential()
    model.add(SwapAxes(1,2),
              CBR(40, 1),
              CBR(40),
              CBR(40),
              nn.MaxPool1D(2),
              CBR(80, 1),
              CBR(80),
              CBR(80),
              nn.MaxPool1D(2),
              CBR(160, 1),
              nn.Dropout(0.3),
              CBR(160),
              CBR(160),
              CBR(160),
              nn.MaxPool1D(2),
              CBR(240, 1),
              nn.Dropout(0.3),
              # CBR(200),
              # CBR(200),
              # CBR(200),
              # nn.MaxPool1D(2),
              # CBR(300, 1)
              )
    for i in range(34):
        model.add(Resnet1D(240))

    model.add(# NCW
              nn.Dropout(0.3),
              nn.Conv1D(3000, 1, 1),
              # NWC
              SwapAxes(1, 2))


def ctc_loss(net, train_features, train_labels):
    preds = net(train_features)
    return loss.CTCLoss()(preds, train_labels)


def get_data_gen(data_dir, str2idx, batch_size=2):
    files = os.listdir(data_dir)
    new_files = []
    for f in files:
        if '.txt' in f:
            new_files.append(f)
    files = new_files
    files = list(set(list(map(lambda f:f.split('.')[0], files))))
    pooling_step = 8
    # np.random.seed(10)
    # while True:
    features = []
    labels = []
    input_len = []
    label_len = []
    np.random.shuffle(files)
    print('start one epoch')
    for idx in range(0, len(files)):
        try:
            feature = np.loadtxt(data_dir+'/'+files[idx]+'.txt') + 1
            #  mfcc.__call__(data_dir+'/'+files[new_idx]+'.wav')
            label = list(open(data_dir+'/'+files[idx]+'.wav.trn').readline().split('\n')[0].replace(' ', ''))
            label = np.array(list(map(lambda l:str2idx[l]+1, label)))
        except Exception as e:
            # print(e, files[idx])
            continue
        features.append(feature)
        labels.append(label)
        input_len.append(len(feature)/pooling_step-pooling_step)
        label_len.append(len(label))
        if len(features) == batch_size:
            maxLenFeature = max(list(map(len, features))) //pooling_step *pooling_step + pooling_step * 2
            maxLenLabel = max(list(map(len, labels)))
            featuresArr = np.zeros([len(features), maxLenFeature, 39], dtype=np.float32)
            labelsArr = np.ones([len(labels), maxLenLabel], dtype=np.float32) * 0  # (len(str2idx)+1)
            for idx in range(len(features)):
                featuresArr[idx, 0:len(features[idx]), :] = np.array(features[idx], dtype=np.float32)
                labelsArr[idx, :len(labels[idx])] = np.array(labels[idx], dtype=np.float32)
            yield featuresArr, labelsArr
            features = []
            labels = []
            input_len = []
            label_len = []


def get_str2idx(data_dir):
    files = os.listdir(data_dir)
    all_words = []
    str2idx = {}
    idx2str = {}
    for f in files:
        if 'trn' in f:
            all_words.extend(list(open(data_dir+'/'+f).readline().split('\n')[0].replace(' ', '')))
    all_words = list(set(all_words))
    for word, idx in enumerate(all_words):
        str2idx[word] = idx
        idx2str[str(idx)] = word
    return str2idx, idx2str


def get_iter(batch_size):
    data_dir = './data'
    train_iter = get_data_gen(data_dir, get_str2idx(data_dir)[1], batch_size)
    for x, y in train_iter:
        yield nd.array(x), nd.array(y)


class ShowProcess():
    """
    process bar to show the process and loss
    """
    i = 0 
    max_steps = 0 
    max_arrow = 50 
    infoDone = 'done'

    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    def show_process(self, loss, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) 
        num_line = self.max_arrow - num_arrow 
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%, loss:' + str(loss) + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0



my_ctcloss = loss.CTCLoss()
def train(net, num_epochs, lr, batch_size):
    with mx.Context(mx.cpu(0)):
        train_ls = []
        trainer = gluon.Trainer(net.collect_params(), 'adam',{
            'learning_rate': lr,
        })
        for epoch in range(num_epochs):
            max_steps = len(os.listdir('./data'))//3 //batch_size
            process_bar = ShowProcess(max_steps, 'OK')
            train_iter = get_iter(batch_size)
            for x, y in train_iter:
                with autograd.record():
                    l = my_ctcloss(net(x), y) # .mean()
                l.backward()
                l = l.mean()
                trainer.step(batch_size)
                train_ls.append(l)
                process_bar.show_process(str(l)[2:8])
            if epoch % 1 == 0:
                # net.save_params('mxnetCnn'+str(epoch)+'.param')
                net.save_parameters('mxnetCnn'+str(epoch)+'.param')
                print('save to', epoch)
        return train_ls


ctx = [ mx.cpu()]
model.initialize(init=init.Xavier(), ctx=ctx)
model.hybridize()
train(model, 10, 0.001, 2)


