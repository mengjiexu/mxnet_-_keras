# mxnet_-_keras
keyword: Mxnet 语音识别 speech recognition keras gluon


## Database
This project is a CNN CTC model written on the gluon interface of mxnet. It is mainly used for speech recognition. The data set used in this project is thchs30.


## Features
MFCC feature is used in audio feature extraction. I changed the feature extraction into np. savetxt before training to facilitate loading. The code for feature extraction is audio.py and audio_utils.py, which are moved from other projects and erased.


## Using mxnet gluon
In this project, the sequential_CNN_CTC.py file is built using the native gluon's Sequntial model, and hybridSequential_CNN_CTC.py is rewritten using HybridSequntial, mainly to improve the efficiency of the GPU.


## Using keras for speech rcognition
The Keras_crnn.py is using CNN and GRU to train the network and the keras_cnn_CTC.py is using deep CNN network to train the network.


## Suggestion
I suggest you to use the HybridSequential_CNN_CTC.py to train the model, because it use resnet-50 and have better performance.

my email:1596446455@qq.com
