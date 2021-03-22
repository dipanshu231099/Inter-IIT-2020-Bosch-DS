import os
import numpy as np
import cv2
from PIL import Image
def num(i):
    i = str(i)
    return '0'*(5-len(i))+i

def combine():
    data=[]
    labels=[]
    test_data =[]
    test_labels =[]
    height = 32
    width = 32
    channels = 3
    classes = 43
    n_inputs = height * width*channels


    n_classes=len(os.listdir("/home/abhishek/django_project4/train"))
    Classes=range(1,n_classes+1)

    for i in Classes:
        #path = "D:/Bosch/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/{0}/".format(num(i))
        path = "/home/abhishek/django_project4/train/class{}".format(i)
        #print(path)
        if os.path.exists(path):
          Class=os.listdir(path)
          n = len(Class)
          n_test = int(n/2)
          for a in Class:

            image=cv2.imread(path+"/"+a)


            image=np.asarray(image)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
            while(n_test>0):
                test_data.append(np.array(size_image))
                test_labels.append(i)



    data=np.array(data)
    data=data.astype('float32')/255
    test_data = np.array(test_data)
    test_data=data.astype('float32')/255


    np.save('/home/abhishek/django_project4/classification/model/new_data.npy', data) # save
    np.save('/home/abhishek/django_project4/classification/model/new_labels.npy', labels) # save
    np.save('/home/abhishek/django_project4/classification/model/new_test_labels.npy', test_labels)
    np.save('/home/abhishek/django_project4/classification/model/new_test_labels.npy', test_data)
#also make test files
