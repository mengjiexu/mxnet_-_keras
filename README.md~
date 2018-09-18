# mxnet_-_keras
mxnet 语音识别 keras gluon 

本项目是使用mxnet的gluon接口写的CNN——CTC模型。主要用于语音识别，本项目使用的数据集是thchs30。

音频特征提取使用的是MFCC特征，我是在训练之前先将特征提取好转换成np.savetxt的形式以方便加载。特征提取的代码是audio.py和audio_utils.py,这两个文件是从其他项目里搬运过来的，侵删。

本项目中my_mxnet.py文件里是使用原生gluon的Sequntial模型搭建的，mx_hybrid是使用HybridSequntial改写的，主要是为了提高GPU运行效率。
