import cv2
import os
import random
from PIL import Image
from augmentations import *

def display():   #give augs list to it
    augs = ['horizontal_shift',
            'brightness',
            'zoom']
    path="/home/abhishek/django_project4/train/"

    sign_class=os.listdir(path)

    for i in sign_class:
        files=os.listdir(path+i)
        im_random=random.choice(files)
        image=Image.open(path+i+"/"+im_random)
        im=cv2.imread(path+i+"/"+im_random)
        os.chdir("/home/abhishek/django_project4/augmented_images")
        n=i+".jpg"
        cv2.imwrite(n,im)
        a=trans(im,augs)

        name="aug"+i+".jpg"
        cv2.imwrite(name,a)



display()
