from django.shortcuts import render, redirect
from django.http import HttpResponse
import os
from .forms import UploadTrainImage
from .forms import UploadTestImage
from .forms import Augmentations
from PIL import Image
from numpy import asarray
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import cv2
from django.core.files.storage import default_storage
from keras.preprocessing.image import load_img
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf

from .augmentations import *
from .retrain_model import *
from .display_images import *
from .plot import *
from plotly.offline import plot
from plotly.graph_objs import Scatter



def index(request):
    None

def addTrainingImage(request):
    form = UploadTrainImage()
    form1 = UploadTestImage()
    if request.method == 'POST':
        if(request.POST.get("form_type")=='test'):
            form1 = UploadTestImage(request.POST, request.FILES)
            image = request.FILES['testing_file']
            if form1.is_valid():
                pred=getpredictions(image)
                return HttpResponse(pred)
            else:
                form1 = UploadTestImage()
                form = UploadTrainImage()
        else:
            form = UploadTrainImage(request.POST, request.FILES)
            files = request.FILES.getlist('img_file')
            if form.is_valid():
                for fname in files:
                    print(fname)
                    trainImageHandler(fname, fname, request.POST['class_name'])
                if(request.POST.get("augmentation")=='yes'):
                    return HttpResponse("done")
                else:
                    return HttpResponse("Succesful")
            else:
                form = UploadTrainImage()
                form1 = UploadTestImage()
    return render(request, 'addTrainingImage.html', {'form':form,'form1':form1})

def trainImageHandler(fname, img, class_name):
    with open('train/'+class_name+'/'+str(fname)+'.jpeg','wb+') as destination:
        print("I am called")
        for chunk in img.chunks():
            destination.write(chunk)

def getpredictions(img):
    with open('temp/ok.jpeg','wb+') as destination:
        for chunk in img.chunks():
            destination.write(chunk)
    img_array=cv2.imread('temp/ok.jpeg')
    im=img_array.astype('float32')/255
    im = cv2.resize(img_array, (32, 32), cv2.INTER_CUBIC)
    im=np.resize(im,(1,32,32,3))
    json_file=open('/home/abhishek/django_project4/classification/model/ii_using_sigmoid.json','r')
    loaded_model_json=json_file.read()
    json_file.close()
    loaded_model=model_from_json(loaded_model_json)
    loaded_model.load_weights("/home/abhishek/django_project4/classification/model/ii_using_sigmoid.h5")
    pred=loaded_model.predict(im)
    return pred


def augment(request):

    request.session['augs']=""
    if request.method == "POST":

        request.session['augs']=request.POST.get('augs')
        if  request.session['token'] == 3:
            display("first",list(request.session['augs'].split(',')))
        elif request.session['token'] == 4:
            display("second",list(request.session['augs'].split(',')))
        elif(request.session['token'] == 2):
            #display("third",['rotation','vertical_flip'])
            display("third",list(request.session['augs'].split(',')))


        return redirect("/app/image/")
    form = Augmentations()
    return render(request,"augmentation.html",{'form':form})

def Merge(request):
    if request.method == 'POST':
        if request.POST.get('aug1')=="first":


            request.session['token'] = 3
            return redirect("/app/augment/")

        if request.POST.get('aug1')=="Second":
            display("second",[])
            request.session['token'] = 4
            return redirect("app/augment/")

    return render (request,"Merge_or_not.html")

def re_train_model(request):

    if(request.session['token'] ==1):
        acc,val_acc,loss,val_loss,m,accu_score=retrain("first",list(request.session['augs'].split(",")))
        plotgraphs(4,acc,val_acc,loss,val_loss,m,accu_score)
        request.session['token'] == 0
        return render(request,"graphs.html")

    elif(request.session['token']==2):
        acc,val_acc,loss,val_loss,m,accu_score=retrain("second",list(request.session['augs'].split(",")))
        plotgraphs(4,acc,val_acc,loss,val_loss,m,accu_score)
        request.session['token'] == 0
        return render(request,"graphs.html")

    elif(request.session['token'] ==3):

        acc,val_acc,loss,val_loss,m,accu_score=retrain("third",list(request.session['augs'].split(",")))
        plotgraphs(4,acc,val_acc,loss,val_loss,m,accu_score)
        request.session['token'] == 0
        return render(request,"graphs.html")


    elif (request.session['token']== 4):

        acc,val_acc,loss,val_loss,m,accu_score=retrain("forth",list(request.session['augs'].split(",")))
        plotgraphs(4,acc,val_acc,loss,val_loss,m,accu_score)
        request.session['token'] == 0
        return render(request,"graphs.html")

    else:
        return HttpResponse("hello world")

def display_images(request):

    if(request.session['token'] == 2):
        images=[]
        original="augmented_images/Orig_classes/original"
        augmented="augmented_images/Orig_classes/augmented"
        im1=os.listdir("/home/abhishek/django_project4/classification/static/augmented_images/Orig_classes/original")
        im2=os.listdir("/home/abhishek/django_project4/classification/static/augmented_images/Orig_classes/augmented")
        for i in range(0,len(im1)):
            a=original+'/'+im1[i]
            b=augmented+'/'+im2[i]
            c=[a,im1[i],b,im2[i]]
            images.append(c)
        return render(request,"dis_org_aug.html",context={'images':images})

    elif(request.session['token'] == 3):
        original="augmented_images/all_classes/original"
        augmented="augmented_images/all_classes/augmented"
        im1=os.listdir("/home/abhishek/django_project4/classification/static/augmented_images/all_classes/original")
        im2=os.listdir("/home/abhishek/django_project4/classification/static/augmented_images/all_classes/augmented")
        for i in range(0,len(im1)):
            a=original+im1[i]
            b=augmented+im2[i]
            c=[a,im1[i],b,im2[i]]
            images.append(c)

    elif(request.session['token'] == 4):

        original="augmented_images/new_classes/original"
        augmented="augmented_images/new_classes/augmented"
        im1=os.listdir("/home/abhishek/django_project4/classification/static/augmented_images/new_classes/original")
        im2=os.listdir("/home/abhishek/django_project4/classification/static/augmented_images/new_classes/augmented")
        for i in range(0,len(im1)):
            a=original+im1[i]
            b=augmented+im2[i]
            c=[a,im1[i],b,im2[i]]
            images.append(c)


        return render(request,"dis_org_aug.html",context={'images':images})

def direct(request):
    if request.method == 'POST':
        if request.POST.get('c1')=="1":


            request.session['token'] = 1
            return redirect("/app/retrain/")

        elif request.POST.get('c1')=="2":

            request.session['token'] = 2
            return redirect("/app/augment/")

        elif request.POST.get('c1')=="3":
            return redirect("/app/")
        elif request.POST.get('c1')=="4":
            return redirect("/app/")

    return render(request,"home.html")


def graphs(request):
    return render(request,"graphs.html")
