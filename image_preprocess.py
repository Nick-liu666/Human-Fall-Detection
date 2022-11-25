import numpy as np
import cv2.cv2 as cv2
import os
import time
class processor:
    def __init__(self,_tw=224,_th=224):
        self.__target_width=_tw
        self.__target_height=_th
    def change_target(self,_tw,_th):
        self.__target_width=_tw
        self.__target_height=_th
    def adjust(self,img):
        # pre-process frames
        # resizing frames
        wn,wh=self.__get_slice(img.shape[1],img.shape[0])
        img=self.__slice_img(img,wn,wh)
        img=self.__resize(img)
        return img

    #1：compute cut part
    def __get_slice(self,_width,_height):
        wscale=_width/self.__target_width
        hscale=_height/self.__target_height
        if wscale<hscale:
            scale=wscale
        else:
            scale=hscale
        #round number
        wn=round(scale*self.__target_width)
        wh=round(scale*self.__target_height)
        return wn,wh

    #2：keep mid part of image，cut exceess part at margin
    def __slice_img(self,img,slicew,sliceh):
        midw=img.shape[1]/2
        midh=img.shape[0]/2
        wl=round(midw-slicew/2)
        wh=round(midw+slicew/2)
        hl=round(midh-sliceh/2)
        hh=round(midh+sliceh/2)
        return img[hl:hh,wl:wh,:]
    
    #3: reise
    def __resize(self,img):
        return cv2.resize(img,(self.__target_width,self.__target_height))

def process(process_path, save_path, pr, isFall):
    for fileName in os.listdir(process_path):
        if fileName == ".DS_Store": continue
        images_path = process_path + os.sep + fileName
        pro_imgs_path = save_path + os.sep + fileName
        if fileName not in os.listdir(save_path):
            os.mkdir(pro_imgs_path)

        # Deal with each image in the one sequence folder
        for img_name in os.listdir(images_path):
            if img_name == "label.txt":
                continue
            img_path = images_path + os.sep + img_name
            img=cv2.imread(img_path)
            img=pr.adjust(img)
            new_img_path = pro_imgs_path + os.sep + img_name
            cv2.imwrite(new_img_path, img)

        # Create the label for each sequence folder
        new_label_path = pro_imgs_path + os.sep + 'label.txt'
        # Check the label file exist or not
        if os.path.exists(new_label_path):
            continue
        # Create the label in the sequence folder
        # 1 for fall, 0 for not fall
        with open(new_label_path, 'x') as f:
            if isFall:
                f.write('1')
            else:
                f.write('0')


if __name__=='__main__':
    pr=processor()
    target_width=224
    target_height=224
    print('image:',end='')
    # img_path=input()
    # fall_path = "/Users/xinyuliu/Desktop/ResNetLSTM-master/RGB/Fall"
    # notFall_path ="/Users/xinyuliu/Desktop/ResNetLSTM-master/RGB/NotFall"
    pro_paths = ["/Users/xinyuliu/Desktop/ResNetLSTM-master/URFD/Fall",
                 "/Users/xinyuliu/Desktop/ResNetLSTM-master/URFD/NotFall",
                 "/Users/xinyuliu/Desktop/ResNetLSTM-master/FDD/Fall",
                 "/Users/xinyuliu/Desktop/ResNetLSTM-master/FDD/NotFall"]
    isFalls = [True,False,True,False]
    # process_path = notFall_path
    # process_path = fall_path
    # save_path = "/Users/xinyuliu/Desktop/ResNetLSTM-master/data"
    save_paths = ["/Users/xinyuliu/Desktop/ResNetLSTM-master/pro_URFD",
                  "/Users/xinyuliu/Desktop/ResNetLSTM-master/pro_URFD",
                  "/Users/xinyuliu/Desktop/ResNetLSTM-master/pro_FDD",
                  "/Users/xinyuliu/Desktop/ResNetLSTM-master/pro_FDD"]


    # Recoding the processing time
    st = time.time()

    print("Start to preprocess images")
    for process_path, isFall, save_path in zip(pro_paths, isFalls, save_paths):
        process(process_path, save_path, pr, isFall)

    # Prnting out the processing time
    print('video preprocessing,time:{:.2f}s'.format(time.time() - st))


    # for fileName in os.listdir(process_path):
    #     if fileName == ".DS_Store": continue
    #     images_path = process_path + os.sep + fileName
    #     pro_imgs_path = save_path + os.sep + fileName
    #     if fileName not in os.listdir(save_path):
    #         os.mkdir(pro_imgs_path)
    #
    #     # Deal with each image in the one sequence folder
    #     for img_name in os.listdir(images_path):
    #         if img_name == "label.txt":
    #             continue
    #         img_path = images_path + os.sep + img_name
    #         img=cv2.imread(img_path)
    #         img=pr.adjust(img)
    #         new_img_path = pro_imgs_path + os.sep + img_name
    #         cv2.imwrite(new_img_path, img)
    #
    #     # Create the label for each sequence folder
    #     new_label_path = pro_imgs_path + os.sep + 'label.txt'
    #     # Check the label file exist or not
    #     if os.path.exists(new_label_path):
    #         continue
    #     # Create the label in the sequence folder
    #     # 1 for fall, 0 for not fall
    #     with open(new_label_path, 'x') as f:
    #         if process_path == fall_path:
    #             f.write('1')
    #         else:
    #             f.write('0')

    # img=cv2.imread(img_path)
    # img=pr.adjust(img)
    # cv2.imshow('abc',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()