# coding:utf-8
import torch
import torchvision as tv
import os
import numpy as np
import random
import cv2.cv2 as cv2
from datetime import datetime
import time
import data_vis

os_file_dir = '/Users/xinyuliu/Desktop/ResNetLSTM-master'

def GetBatch(data, label, sampleNum=8, batchnum=8):
    for i in range(0, sampleNum // batchnum):
        low = i * batchnum
        x = data[low:low + batchnum]
        y = label[low:low + batchnum]
        yield x, y


class net:
    def __init__(self, hidden=128, lr=0.0001, data_name='URFD'):
        # features=4608
        self.data_name = data_name
        self.hidden = hidden
        # self.features = 4608
        # self.features = 18432
        self.features = 100352
        # features=512
        layers = 2
        output = 1
        self.frames = 30
        self.sampleNum = -1
        # Pre-trained weight and parameters
        self.cnn = tv.models.resnet50(pretrained=True)
        # self.cnn = tv.models.resnet50(weights = tv.models.ResNet50_Weights.IMAGENET1K_V2)
        self.cnn.eval()
        self.final_pool = torch.nn.MaxPool2d(3, 2)

        self.LSTM = torch.nn.LSTM(self.features, hidden, layers, batch_first=True)
        self.Linear = torch.nn.Linear(hidden, output)
        self.criteria = torch.nn.MSELoss()
        self.opt = torch.optim.Adam([{'params': self.LSTM.parameters()},
                                     {'params': self.Linear.parameters()}], lr)
        self.data = None
        self.label = None

    def loadData(self, samplePath=None):
        self.picRead(samplePath)
        self.normalize()
        self.extractFeature()
        self.shuffle()

    def picRead(self, dirpath=None):
        '''
        Read frames and store in list
        '''
        if dirpath is None:
            # dirpath=os.path.dirname(__file__)+os.sep+'sample'
            # dirpath = os.path.dirname(__file__) + os.sep + 'pro_' + self.data_name
            dirpath = os_file_dir + os.sep + 'pro_' + self.data_name

        st = time.time()
        data = []
        label = []
        sampleNum = 0
        for sname in os.listdir(dirpath):
            if sname == ".DS_Store":
                continue
            spath = dirpath + os.sep + sname
            frames = []

            # Get the correct order path name
            lis_dir = os.listdir(spath)
            lis_dir.remove("label.txt")
            # lis_dir.sort(key=lambda x: x[:-4])

            sum = 0
            accum = len(lis_dir) / self.frames
            for i in range(1, self.frames + 1):
                # imgname='o ({}).jpg'.format(i)
                imgname = lis_dir[int(sum)]
                sum += accum
                img = cv2.imread(spath + os.sep + imgname)
                frames.append(img)
            data.append(frames)

            labelPath = spath + os.sep + 'label.txt'
            tx = open(labelPath)
            str1 = tx.read()
            tx.close()
            # label.append([float(str1)])
            label.append([int(str1)])

            sampleNum += 1
            print('sample{} finished'.format(sampleNum))
        print('sample loaded,time:{:.2f}s'.format(time.time() - st))
        self.sampleNum = sampleNum
        self.data = np.array(data)
        self.label = np.array(label)

    def normalize(self):
        '''
        normalize the data value
        Using functions from torchvision
        '''
        data = self.data
        label = self.label

        st = time.time()
        print('normalization start')
        sampleNum = self.sampleNum
        frames = self.frames
        ndata = torch.zeros(sampleNum, frames, 3, 224, 224)
        for s in range(sampleNum):
            for f in range(frames):
                img = data[s][f]
                transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))
                ])
                img = transform(img)
                img = torch.autograd.Variable(img, requires_grad=True)
                ndata[s][f] = img

        # nlabel=label/15
        nlabel = label
        nlabel = torch.Tensor(nlabel)

        print('normalization finished,time:{:.2f}s'.format(time.time() - st))

        self.data = ndata
        self.label = nlabel

    def extractFeature(self):
        '''
        ResNet 50 extract feature maps from frames
        flatten them in 1 dimension
        save in self.data
        '''
        st = time.time()
        print('feature extracting start')
        n = self.cnn
        pool = self.final_pool
        data = self.data
        sampleNum = self.sampleNum
        frames = self.frames
        ndata = torch.zeros(sampleNum, frames, self.features)
        # ndata=torch.zeros(sampleNum,frames,4608)
        # ndata=torch.zeros(sampleNum,frames,512)
        with torch.no_grad():
            for i in range(sampleNum):
                input = data[i]
                x = n.conv1(input)
                x = n.bn1(x)
                x = n.relu(x)
                x = n.maxpool(x)
                x = n.layer1(x)
                x = n.layer2(x)
                x = n.layer3(x)
                x = n.layer4(x)

                # x = pool(x)
                # x=n.avgpool(x)
                x = x.flatten(start_dim=1)
                ndata[i] = x
        self.data = ndata
        print('feature extracted,time:{:.2f}s'.format(time.time() - st))

    def shuffle(self):
        '''
        Shuffle the data with labels
        '''
        st = time.time()
        indices = np.arange(self.sampleNum)
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.label = self.label[indices]
        print('shuffle,time:{:.2f}s'.format(time.time() - st))

    def train(self, epochNum=50, batchNum=8, finalLoss=1e-5):
        '''
        Training with data
        '''
        # Check data existence
        if self.data is None:
            self.loadData()
            data = self.data
            label = self.label
        else:
            data = self.data
            label = self.label

        # Set 80% train and 20% validation set.
        sampleNum = len(label)
        num_test = int(0.2 * sampleNum)
        train_input = data[num_test:]
        train_output = label[num_test:]
        test_input = data[:num_test]
        test_output = label[:num_test]
        trainNum = sampleNum - num_test
        print("Train number: ", trainNum, ", Batch number", batchNum)
        if trainNum < batchNum:
            raise Exception('samples are not enough，or decrease batch size')

        self.LSTM.train()
        self.Linear.train()

        print('Start to train')
        # savedir = os.path.dirname(__file__) + os.sep + 'save_' + self.data_name + os.sep
        # Check path is exit or not
        savedir = os_file_dir + os.sep + 'save_' + self.data_name + os.sep
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        tr_loss_lis = []
        te_loss_lis = []

        tr_num_correct_lis = []
        te_num_correct_lis = []

        # Stat to train with 50 epochs
        for epoch in range(epochNum):
            train_loss = 0
            test_loss = 0
            tr_num_correct = 0
            te_num_correct = 0

            for x, y in GetBatch(train_input, train_output,
                                 trainNum, batchNum):
                self.opt.zero_grad()
                out, _ = self.LSTM(x)
                out_last = out[:, -1, :]
                pred = self.Linear(out_last)
                loss = torch.sqrt(self.criteria(pred, y))
                loss.backward()
                self.opt.step()

                train_loss += loss.item()

                # Computing the correct prediction numbers
                train_pred = torch.round(pred)

                tr_num_correct += torch.eq(train_pred, y).sum().float().item()

            train_loss /= trainNum // batchNum

            # Validation loss computation
            with torch.no_grad():
                out, _ = self.LSTM(test_input)
                out_last = out[:, -1, :]
                pred = self.Linear(out_last)
                test_loss = torch.sqrt(self.criteria(pred, test_output))

                test_pred = torch.round(pred)
                te_num_correct = torch.eq(test_pred, test_output).sum().item()

            print('epoch:{},train:{},test:{}'.format(
                epoch, train_loss, test_loss))

            # save the loss for graph
            tr_loss_lis.append(train_loss)
            te_loss_lis.append(test_loss)
            tr_num_correct_lis.append(tr_num_correct / len(train_output))
            te_num_correct_lis.append(te_num_correct / len(test_output))

            # Save the trained parameter and weight
            if (epoch % 20 == 0) or (test_loss < finalLoss):
                state = {'net1': self.LSTM.state_dict(),
                         'net2': self.Linear.state_dict(),
                         'optimizer': self.opt.state_dict()}
                saveName = '{}.pth'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
                torch.save(state, savedir + saveName)

                if test_loss < finalLoss:
                    break
        # Draw all loss, accuracy graph
        data_vis.draw_fig(tr_loss_lis, te_loss_lis, "loss", epochNum,
                          data_name=self.data_name + str(self.hidden))
        data_vis.draw_fig(tr_num_correct_lis, te_num_correct_lis, "acc", epochNum,
                          data_name=self.data_name + str(self.hidden))

    def eval(self, samplePath):
        '''
        Classify samples by trained model
        '''
        self.LSTM.eval()
        self.Linear.eval()
        lis_dir = os.listdir(samplePath)
        lis_dir.remove("label.txt")
        sum = 0
        accum = len(lis_dir) / self.frames
        with torch.no_grad():
            # print('Start to load')
            print("Test the ", samplePath)
            sample = torch.zeros(30, 3, 224, 224)
            for j in range(1, self.frames + 1):
                # imgPath=samplePath+os.sep+'o ({}).jpg'.format(j)

                imgname = lis_dir[int(sum)]
                sum += accum
                imgPath = samplePath + os.sep + imgname

                img = cv2.imread(imgPath)
                img = self.__preprocess(img)
                sample[j - 1] = img
            sample = self.__getFeature(sample)
            sample = sample.flatten(start_dim=1)
            sample = sample.unsqueeze(dim=0)
            # print('load success')

            out, _ = self.LSTM(sample)
            out_last = out[:, -1, :]
            pred = self.Linear(out_last)

        pred = pred
        pred = pred.data.cpu().numpy()[0][0]
        labelPath = samplePath + os.sep + 'label.txt'
        tx = open(labelPath)
        str1 = tx.read()
        print('pred:{0},truth:{1}'.format(pred, str1))
        if int(round(pred)) != int(str1):
            print('Prediction error.....')
            print('Prediction error.....')
            print('Prediction error.....')
        return int(round(pred)), int(str1)

    def predict(self, fold_path='/Users/xinyuliu/Desktop/ResNetLSTM-master/eval_'):
        fold_path = fold_path + self.data_name
        predict_list = []
        label_list = []
        for sampleName in os.listdir(fold_path):
            if sampleName == '.DS_Store': continue  # Skip Auto generated file
            sample_path = fold_path + os.sep + sampleName
            pred, label = self.eval(sample_path)
            predict_list.append(pred)
            label_list.append(label)
        print("predict List:", predict_list)
        print("label_list", label_list)
        # conMatrix = data_vis.confusionM(np.array(predict_list), np.array(label_list))
        conMatrix = data_vis.confusionM(predict_list, label_list)
        data_vis.plot_confusion_matrix(conMatrix, classes=('Not Fall', 'Fall'),
                                       data_name=self.data_name + str(self.hidden))
        print("conMatrix: ", conMatrix)

    def load(self, saveName):
        '''
        load weight:path is \\save\\saveName
        '''
        # save_dir = os.path.dirname(__file__) + os.sep + 'save_' + self.data_name
        save_dir = os_file_dir + os.sep + 'save_' + self.data_name
        savePath = save_dir + os.sep + saveName
        checkpoint = torch.load(savePath)
        self.LSTM.load_state_dict(checkpoint['net1'])
        self.Linear.load_state_dict(checkpoint['net2'])
        self.opt.load_state_dict(checkpoint['optimizer'])

    def __preprocess(self, img):
        '''
        single frame normalization, only used in the evaluation
        '''
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ])
        img = transform(img)
        img = torch.autograd.Variable(img, requires_grad=True)
        return img

    def __getFeature(self, input):
        '''
        Single frame feature extract, only used in evaluation
        '''
        n = self.cnn
        pool = self.final_pool
        with torch.no_grad():
            x = n.conv1(input)
            x = n.bn1(x)
            x = n.relu(x)
            x = n.maxpool(x)
            x = n.layer1(x)
            x = n.layer2(x)
            x = n.layer3(x)
            x = n.layer4(x)
            # x = pool(x)
            # x=n.avgpool(x)
        return x



if __name__ == '__main__':
    num_hiddens = [64, 128, 256, 512, 1024]
    datasets = ['URFD', 'FDD']
    temp_n = net(data_name=datasets[0])
    temp_n.loadData()
    URDF_data, URDF_label = temp_n.data, temp_n.label

    temp_n = net(data_name=datasets[1])
    temp_n.loadData()
    FDD_data, FDD_label = temp_n.data, temp_n.label

    for data in datasets:
        for num_hidden in num_hiddens:
            n = net(hidden = num_hidden, data_name= data)
            if data == datasets[0]:
                n.data, n.label = URDF_data, URDF_label
            else:
                n.data, n.label = FDD_data, FDD_label
            n.train()
            n.predict()

'''
import ResNetLSTM as rnl
n=rnl.net()
n.loadData()
n.load('2022-08-23-10-11-26.pth')
n.train() #Start to train
n.eval('/Users/xinyuliu/Desktop/ResNetLSTM-master/eval_URFD/fall-29-cam0-rgb')

Store the data in varible
a,b=n.data,n.label

modified code, need reload 
import importlib
importlib.reload(rnl)
n.data,n.label=a,b
'''