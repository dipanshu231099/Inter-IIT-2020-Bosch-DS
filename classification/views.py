from django.shortcuts import render
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

from plotly.offline import plot
from plotly.graph_objs import Scatter


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



def graphs(request):
    x_data = [0,1,2,3]
    y_data = [x**2 for x in x_data]
    plot_div = plot([Scatter(x=x_data, y=y_data,
                        mode='lines', name='test',
                        opacity=0.8, marker_color='green')],
               output_type='div')
    return render(request, "graphs.html", context={'plot_div': plot_div})

def augment(request):
    return render(request,"augmentation.html")
