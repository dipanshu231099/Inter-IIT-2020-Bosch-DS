from django.shortcuts import render
from django.http import HttpResponse
import os
from .forms import UploadTrainImage 
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
from .forms import UploadTrainImage
from plotly.offline import plot
from plotly.graph_objs import Scatter


def addTrainingImage(request):
    if request.method == 'POST':
        form = UploadTrainImage(request.POST, request.FILES)
        if form.is_valid():
            trainImageHandler(request.FILES['img_file'], request.POST['class_name'])
            getpred=getpredictions(request.FILES['img_file'])

            
            return HttpResponse(getpred)
    else:
        form = UploadTrainImage()
    return render(request, 'addTrainingImage.html', {'form':form})

def trainImageHandler(img, class_name):
    with open('train/'+class_name+'/'+class_name+'.jpeg','wb+') as destination:
        for chunk in img.chunks():
            destination.write(chunk)


def testImage():
    pass


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



def index(request):
    x_data = [0,1,2,3]
    y_data = [x**2 for x in x_data]
    plot_div = plot([Scatter(x=x_data, y=y_data,
                        mode='lines', name='test',
                        opacity=0.8, marker_color='green')],
               output_type='div')
    return render(request, "index.html", context={'plot_div': plot_div})
