import os
import matplotlib.pyplot as plt
import numpy as np
import itertools

class visualize:
    def __int__(self):
        self.data_path = "/Users/xinyuliu/Desktop/ResNetLSTM-master/data"

    def avg(self):
        lis = []
        length = []
        for lisdir in os.listdir(self.data_path):
            if lisdir == ".DS_Store": continue
            images_path = self.data_path + os.sep + lisdir
            length.append(len(os.listdir(images_path)))
        nplength = np.array(length)
        npsecond = np.round(np.array(length) / 30)
        print(nplength)
        print(npsecond)
        return np.min(nplength),np.average(nplength),np.max(nplength)

# Draw loss or accurcy graph.
def draw_fig(list,list2,name,epoch, data_name = "URFD"):
    x1 = range(1, epoch+1)
    # print(x1)
    y1 = list
    y2 = list2

    plt.cla()
    plt.title(name + ' vs. epoch', fontsize=15)
    plt.plot(x1, y1, '.-', label='train_' + name)
    plt.plot(x1, y2, '.-', label="val_" + name)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(name, fontsize=15)
    plt.grid()
    plt.legend()
    plt.savefig("./graph/"+name+"_tend_" + data_name +".png")
    plt.show()

# output the confusion Matrix with list type
def confusionM(preds, labels):
    cmt = np.zeros((2,2))
    for pl, tl in zip(preds, labels):
        cmt[tl, pl] = cmt[tl, pl] + 1
    return cmt

# plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, data_name="URFD"):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # fmt = '.2f' if normalize else 'd'
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./graph/confusion_matrix_"+data_name+".png")
    plt.show()

# correctly predict number / incorrectly predict number
# precision = (TP + TN)/(TP + FP + FP + FN)
def precision(con_matrix):
    correct_predict = 0.0
    for i in range(len(con_matrix[0])):
        correct_predict += con_matrix[i][i]
    return correct_predict / np.sum(con_matrix)
    pass

# Recall rate = TP/(TP +FN)
def recall(con_matrix,index=0):
    # print(np.sum(con_matrix,axis=1),'recal')
    if np.sum(con_matrix,axis=1)[index] == 0: return con_matrix[index][index]
    return con_matrix[index][index] / np.sum(con_matrix,axis=1)[index]
    pass

def specif(con_matrix,index=1):
    # print(np.sum(con_matrix,axis=1),'recal')
    if np.sum(con_matrix, axis=1)[index] == 0: return con_matrix[index][index]
    return con_matrix[index][index] / np.sum(con_matrix, axis=1)[index]
    pass

if __name__=='__main__':
    pass
    # a = os.listdir('/Users/xinyuliu/Desktop/ResNetLSTM-master/data/adl-08-cam0-rgb')
    # a.sort(key=lambda x:x[:-4])
    # for b in a:
    #     print(b)

    # draw_fig([1,2,3],[2,3,4],"loss",3)
    # vis = visualize()
    # vis.__int__()
    # print(vis.avg())
