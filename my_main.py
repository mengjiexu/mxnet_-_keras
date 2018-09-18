import os
import numpy as np
# import scipy.io.wavfile as wav
from collections import Counter
# from python_speech_features import mfcc
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers import Lambda, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import add, concatenate
from keras import backend as K
from keras.optimizers import SGD, Adadelta, Adam
from keras.layers.recurrent import GRU
from keras.preprocessing.sequence import pad_sequences
import sys
from keras.layers import BatchNormalization
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./preprocessing'))
from audio import MFCC


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

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




def get_data_gen(data_dir, str2idx, batch_size=2):
    print(str2idx)
    mfcc = MFCC()
    files = os.listdir(data_dir)
    new_files = []
    for f in files:
        if '.txt' in f:
            new_files.append(f)
    files = new_files
    files = list(set(list(map(lambda f:f.split('.')[0], files))))
    while True:
        np.random.shuffle(files)
        print('start one epoch')
        for idx in range(0, len(files), batch_size):
            try:
                features = []
                labels = []
                input_len = []
                label_len = []
                for new_idx in range(idx, idx+batch_size):
                    feature = np.loadtxt(data_dir+'/'+files[new_idx]+'.txt')
                    #  mfcc.__call__(data_dir+'/'+files[new_idx]+'.wav')
                    label = list(open(data_dir+'/'+files[new_idx]+'.wav.trn').readline().split('\n')[0].replace(' ', ''))
                    label = np.array(list(map(lambda l:str2idx[l]+1, label)))
                    features.append(feature)
                    labels.append(label)
                    input_len.append(len(feature)/4-2)
                    label_len.append(len(label))
                maxLenFeature = max(list(map(len, features))) //4 *4 + 8
                maxLenLabel = max(list(map(len, labels)))
                featuresArr = np.zeros([batch_size, maxLenFeature, 39], dtype=np.float32)
                labelsArr = np.ones([batch_size, maxLenLabel], dtype=np.float32) * 0  # (len(str2idx)+1)
                for idx in range(batch_size):
                    featuresArr[idx, 0:len(features[idx]), :] = np.array(features[idx], dtype=np.float32)
                    labelsArr[idx, :len(labels[idx])] = np.array(labels[idx], dtype=np.float32)
                input_len = np.array(input_len, dtype=np.int64)
                label_len = np.array(label_len, dtype=np.int64)
                featuresArr = {
                    'features':featuresArr,
                    'labels':labelsArr,
                    'input_len':input_len,
                    'label_len':label_len
                }
                # print(input_len, label_len)
                yield featuresArr, np.ones([batch_size, 2])
            except Exception as e:
                print(e)
                continue


# 利用backend调用ctc
def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def createModel(label_num = 3000):
    beishu = 2
    input_data = Input(name='features', shape=(None, 39))
    x1 = Conv1D(40 * beishu, 1, padding='same',activation='relu')(input_data)
    x1 = Conv1D(40 * beishu, 3, padding='same',activation='relu')(x1)
    x1 = Conv1D(40 * beishu, 3, padding='same',activation='relu')(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Conv1D(80 * beishu, 1, padding='same',activation='relu')(x1)
    x1 = Conv1D(80 * beishu, 3, padding='same',activation='relu')(x1)
    x1 = Conv1D(80 * beishu, 3, padding='same',activation='relu')(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Conv1D(160 * beishu, 1, padding='same',activation='relu')(x1)
    x1 = Conv1D(160 * beishu, 3, padding='same',activation='relu')(x1)
    x1 = Conv1D(160 * beishu, 3, padding='same',activation='relu')(x1)
    x1 = Conv1D(160 * beishu, 3, padding='same')(x1)
    # x1 = MaxPooling1D(1)(x1)
    x1 = BatchNormalization()(x1)
    x1 = keras.layers.ReLU()(x1)
    # with tf.device('/cpu'):
    x1 = GRU(100, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', dropout=0.3)(x1)
    x2 = GRU(100, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', dropout=0.3)(x1)
    x3 = add([x1, x2])
    x4 = Dense(200, activation='relu')(x3)
    x4 = Dropout(0.3)(x4)
    preds = Dense(label_num)(x4)
    preds = Activation('softmax', name='Activation0')(preds)
    model_data = Model(input_data, outputs=preds)

    # ctc
    labels = Input(name='labels', shape=[None], dtype='float32')
    input_len = Input(name='input_len', shape=[1], dtype='int64')
    label_len = Input(name='label_len', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')([labels, preds, input_len, label_len])

    model = Model(inputs=[input_data, labels, input_len, label_len],
                  outputs=loss_out)
    model.summary()
    adam = Adam(lr=0.01)
    model.compile(loss={'ctc':lambda y_true, output:output}, optimizer=adam)

    print('model compile over')
    return model, model_data


data_dir = '/media/xmj/ubt_2t/中文语音识别/data_thchs30/data'
train_iter = get_data_gen(data_dir, get_str2idx(data_dir)[1], 80)
# data, label = next(train_iter)
# print(data, label)
model, model_data = createModel(len(get_str2idx(data_dir)[0])+2)

# check point
checkpointer = keras.callbacks.ModelCheckpoint(filepath="crrnn-{epoch:02d}.hdf5", save_best_only=False, verbose=1,
                                               period=1)
model.load_weights('crrnn-04.hdf5')
model.fit_generator(train_iter, steps_per_epoch=len(os.listdir(data_dir))//(2*4)//20, epochs=100,
                  callbacks=[checkpointer], use_multiprocessing=True, workers=2)

















