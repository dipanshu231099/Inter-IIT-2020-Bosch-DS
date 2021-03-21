from django.shortcuts import render, redirect
from django.http import HttpResponse
import os
from .forms import UploadTrainImage
from .forms import UploadTestImage
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
                    return HttpResponse("aaaaaaaaaaaaaaa")
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
    return render(request,"augmentation.html")

def Merge(request):
    if request.method == 'POST':
        if request.POST.get('aug1')=="first":

            display("first",[])
            request.session['token'] = 1
            return redirect("/app/dis_org_aug/")

        if request.POST.get('aug1')=="Second":
            display("second",[])
            request.session['token'] = 2
            return redirect("/dis_org_aug/")

    return render (request,"Merge_or_not.html")








def re_train_model(request):
    request.session['token'] ==1  #to be changed
    if(request.session['token'] ==1):


        a,b=retrain(1,[])
        plot_div = graphs(a,b)

        return render(request,"graphs.html", context={'plot_div': plot_div})

    elif (request.session['token']== 2):
        return HttpResponse("its 2")
    else:
        return HttpResponse("hello world")

def display_images(request):
    request.session['token'] =1  #to be changed
    if(request.session['token'] ==1):
        images=[]
        original="/home/abhishek/django_project4/classification/static/augmented_images/new_classes/original"
        augmented="/home/abhishek/django_project4/classification/static/augmented_images/new_classes/augmented"
        im1=os.listdir(original)
        im2=os.listdir(augmented)
        for i in range(0,len(im1)):
            a=original+im1[i]
            b=augmented+im2[i]
            c=[a,b]
            images.append(c)


        return render(request,"dis_org_aug.html",context={'images':images})
    else:
        images=[]
        original="/home/abhishek/django_project4/classification/static/augmented_images/all_classes/original"
        augmented="/home/abhishek/django_project4/classification/static/augmented_images/all_classes/augmented"
        im1=os.listdir(original)
        im2=os.listdir(augmented)
        for i in range(0,len(im1)):
            a=original+im1[i]
            b=augmented+im2[i]
            c=[a,b]
            images.append(c)


        return render(request,"dis_org_aug.html",context={'images':images})

def direct(request):
    return render(request,"retrain.html")
