# Human Fall Detection

## 文件介绍  
ResNetLSTM.py: 主程序 Resnet50 + LSTM train

image_preprocess.py:  处理图片转换成224x224的大小，居中剪切  

video_preprocess.py: 将视频数据提取成图片帧数，用于ResNet50的输入

data_vis.py：画loss trend， accuracy trend， confusion matrix的图

data: 数据存放文件夹

weight: 训练过的参数存档 

graph: 储存生成的loss， accuracy trend和confusion matirx的图像

# ResNetLSTM

用训练好的ResNet50 网络从每一帧图像输入来提取对应feature map，之后在把连续的feature map作为一个sample输入LSTM 网络中进行训练，得出分类的结果。
