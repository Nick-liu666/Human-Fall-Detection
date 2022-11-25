import cv2.cv2 as cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time

# Convert video into frames format
class video_preprocess:
    def __init__(self, dirPath="/Users/xinyuliu/Desktop/902 Project/data/FallDataset/Home_01/Home_01/Videos",
                savePath='/Users/xinyuliu/Desktop/ResNetLSTM-master/fallDataset/Fall',
                roomName='Home_01'):
        self.dirPath = dirPath
        self.pathDir = os.listdir(dirPath)
        self.savePath = savePath
        self.roomName = roomName

    # print out the detail information about video samples
    def detail(self):
        f = []
        for dirName in self.pathDir:
            if dirName == ".DS_Store":
                continue
            videoPath = self.dirPath + '/' + dirName

            vc = cv2.VideoCapture(videoPath)
            if vc.isOpened():
                success, frame = vc.read()
            else:
                success = False
            fps = vc.get(cv2.CAP_PROP_FPS)
            frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
            print("fps", fps)
            print("Frames Count", frames)
            f.append(frames)
        return f


    def frames(self):
        index = 0
        for dirName in self.pathDir:
            # For avoiding error for .DS_Store
            if dirName == ".DS_Store":
                continue

            videoPath = self.dirPath + '/' + dirName
            print("videospath:", videoPath)

            vc = cv2.VideoCapture(videoPath)

            if vc.isOpened():
                success, frame = vc.read()
                # print(True)
            else:
                success = False
                # print(False)

            # fps = vc.get(cv2.CAP_PROP_FPS)
            # size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
            #         int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            times = vc.get(cv2.CAP_PROP_FRAME_COUNT)
            # print("fps", fps)
            # print("size", size)
            print("Frames Count", times)

            # Set the save path for each video
            # Check the folder exit or not
            newSavePath = self.savePath + '/' + self.roomName + str(index)
            if self.roomName + str(index) not in os.listdir(self.savePath):
                os.mkdir(newSavePath)

            # For next folder to save generated images
            index+=1

            c = 0
            frameRate = 5  # skip frames rate（pick one frames after 5 frames skip）
            frame_index = 1
            while success:
                success, frame = vc.read()
                if success:
                    if (c % frameRate ==0):
                        # Save the image in local
                        cv2.imwrite(newSavePath + '/' +  str(frame_index) + '.jpg', frame)
                        frame_index += 1
                    c += 1
                    cv2.waitKey(1)
            vc.release()
        print("All done")

if __name__ == "__main__":


    dirPaths = ['/Users/xinyuliu/Desktop/902 Project/data/FallDataset/Home_01/Home_01/Videos',
                '/Users/xinyuliu/Desktop/902 Project/data/FallDataset/Home_02/Home_02/Videos',
                '/Users/xinyuliu/Desktop/902 Project/data/FallDataset/Coffee_room_01/Coffee_room_01/Videos',
                '/Users/xinyuliu/Desktop/902 Project/data/FallDataset/Coffee_room_02/Coffee_room_02/Videos',
                '/Users/xinyuliu/Desktop/902 Project/data/FallDataset/Office_01/Office_01',
                '/Users/xinyuliu/Desktop/902 Project/data/FallDataset/Office_02/Office_02',
                '/Users/xinyuliu/Desktop/902 Project/data/FallDataset/Lecture_room_01/Lecture_room_01',
                '/Users/xinyuliu/Desktop/902 Project/data/FallDataset/Lecture_room_02/Lecture_room_02']
    dirPaths2 = ["/Users/xinyuliu/Desktop/902 Project/data/UR Fall Detection Dataset/Video/NonFall",
                 "/Users/xinyuliu/Desktop/902 Project/data/UR Fall Detection Dataset/Video/Fall"]
    f_list = []
    for path in dirPaths:
        vp = video_preprocess(path)
        f_list.extend(vp.detail())
    print(f_list)
    print("max frame: ", max(f_list))
    print("min frame: ", min(f_list))
    print("ave frames", sum(f_list)/len(f_list))
    f_list = sorted(f_list)
    f_list = np.array(f_list)
    x = [a * max(f_list)/14 for a in range(15)]
    y = np.zeros(len(x))
    for f in f_list:
        for i in range(len(x)):
            if f > x[i] and f <= x[i+1]:
                y[i] += 1

    # result = np.unique(f_list, return_index=True, return_counts=True, return_inverse=True)
    # print(result)
    # x = result[0]
    # y = result[3]

    # Generate graph
    plt.plot(x, y, 'go:', label='frame number', linewidth=2)  # green, point, dotted line, label, line width

    plt.ylabel('num of samples')
    plt.xlabel('num of frames')
    plt.legend()  # show the legend
    plt.grid()  # show grid

    #  show graph
    plt.show()
    #
    # roomNames = ['Home_01','Home_02',
    #              'Coffee_room_01','Coffee_room_02',
    #              'Office_01','Office_02',
    #              'Lecture_room_01','Lecture_room_02']
    #
    # savePathFall = '/Users/xinyuliu/Desktop/ResNetLSTM-master/fallDataset/Fall'
    # savePathNotFall = '/Users/xinyuliu/Desktop/ResNetLSTM-master/fallDataset/NotFall'
    # savePaths = [savePathFall,savePathNotFall,
    #              savePathFall,savePathNotFall,
    #              savePathFall,savePathNotFall,
    #              savePathFall,savePathNotFall,]
    #
    # # Recoding the processing time
    # st = time.time()
    #
    # # Run the video process
    # for dirPath, roomName, savePath in zip(dirPaths,roomNames,savePaths):
    #     vp = video_preprocess(dirPath,savePath,roomName)
    #     vp.frames()
    #
    # # Prnting out the processing time
    # print('video preprocessing,time:{:.2f}s'.format(time.time() - st))