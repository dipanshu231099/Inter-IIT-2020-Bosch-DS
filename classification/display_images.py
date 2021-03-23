import cv2
import os
import random
from PIL import Image
from .augmentations import *

def display(condition,augs):   #give augs list to it

    if(condition=="first"): #add augmentations and then merge with the training data

        path="/home/abhishek/django_project4/train/"

        sign_class=os.listdir(path)

        for i in sign_class:
            files=os.listdir(path+i)
            im_random=random.choice(files)
            image=Image.open(path+i+"/"+im_random)
            im=cv2.imread(path+i+"/"+im_random)
            os.chdir("/home/abhishek/django_project4/classification/static/augmented_images/new_classes/original")
            n=i+".jpg"
            cv2.imwrite(n,im)
            a=trans(im,augs)
            os.chdir("/home/abhishek/django_project4/classification/static/augmented_images/new_classes/augmented")
            name="aug"+i+".jpg"
            cv2.imwrite(name,a)

    elif(condition=="second"):
        path="/home/abhishek/django_project4/train/"
        path2="/home/abhishek/django_project4/Orig_train/"
        sign_class=os.listdir(path)
        sign_class2=os.listdis(path2)
        for i in sign_class:
            files=os.listdir(path+i)
            im_random=random.choice(files)
            image=Image.open(path+i+"/"+im_random)
            im=cv2.imread(path+i+"/"+im_random)
            os.chdir("/home/abhishek/django_project4/classification/static/augmented_images/all_classes/original")
            n=i+".jpg"
            cv2.imwrite(n,im)
            a=trans(im,augs)
            os.chdir("/home/abhishek/django_project4/classification/static/augmented_images/all_classes/augmented")

            name="aug"+i+".jpg"
            cv2.imwrite(name,a)
        for j in sign_class:
            files=os.listdir(path2+j)
            im_random=random.choice(files)
            image=Image.open(path+j+"/"+im_random)
            im=cv2.imread(path+j+"/"+im_random)
            os.chdir("/home/abhishek/django_project4/classification/static/augmented_images/all_classes")
            n=j+".jpg"
            cv2.imwrite(n,im)
            a=trans(im,augs)
            os.chdir("/home/abhishek/django_project4/classification/static/augmented_images/all_classes")
            name="aug"+j+".jpg"
            cv2.imwrite(name,a)


    elif(condition == "third"):

            path="/home/abhishek/django_project4/Orig_train/"

            sign_class=os.listdir(path)

            for i in sign_class:
                files=os.listdir(path+i)
                im_random=random.choice(files)
                image=Image.open(path+i+"/"+im_random)
                im=cv2.imread(path+i+"/"+im_random)
                os.chdir("/home/abhishek/django_project4/classification/static/augmented_images/Orig_classes/original")
                n=i+".jpg"
                cv2.imwrite(n,im)
                a=trans(im,augs)
                os.chdir("/home/abhishek/django_project4/classification/static/augmented_images/Orig_classes/augmented")
                name="aug"+i+".jpg"
                cv2.imwrite(name,a)


