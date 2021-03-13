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

def addTrainingImage(request):
    if request.method == 'POST':
        form = UploadTrainImage(request.POST, request.FILES)
        if form.is_valid():
            trainImageHandler(request.FILES['img_file'], request.POST['class_name'])
            return HttpResponse("Hola this was succesful")
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
    img_array=asarray(img)
    img_array=img_array.astype('float')/255
    json_file=open('/home/abhishek/django_project4/classification/model/ii.json','r')
    loaded_model_json=json_file.read()
    json_file.close()
    loaded_model=model_from_json(loaded_model_json)
    loaded_model.load_weights("/home/abhishek/django_project4/classification/model/ii.h5")
    pred=loaded_model.predict(img_array,axis=1)
    return pred


