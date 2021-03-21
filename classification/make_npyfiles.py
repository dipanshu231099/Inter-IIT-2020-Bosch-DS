import os
import numpy as np
import cv2
def num(i):
    i = str(i)
    return '0'*(5-len(i))+i

def combine():
    data=[]
    labels=[]

    height = 32
    width = 32
    channels = 3
    classes = 43
    n_inputs = height * width*channels

   
    n_classes=len(os.listdir("/home/abhishek/django_project4/train"))
    Classes=range(0,n_classes)

    for i in Classes:
        #path = "D:/Bosch/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/{0}/".format(num(i))
        path = "/home/abhishek/django_project4/train"
        #print(path)
        if os.path.exists(path):
          Class=os.listdir(path)
          for a in Class:
            try:
                image=cv2.imread(path+a)
                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((height, width))
                data.append(np.array(size_image))
                labels.append(i)

            except AttributeError:
                print('N')
    data=data.astype('float32')/255
    np.save('/home/abhishek/django_project4/classification/model/new_data.npy', data) # save
    np.save('/home/abhishek/django_project4/classification/model/new_labels.npy', labels) # save

#also make test files