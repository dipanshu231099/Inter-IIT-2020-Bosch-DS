from django.shortcuts import render
from django.http import HttpResponse
import os
from .forms import UploadTrainImage 

def addTrainingImage(request):
    if request.method == 'POST':
        form = UploadTrainImage(request.POST, request.FILES)
        if form.is_valid():
            trainImageHandler(request.FILES['img_file'], request.POST['class_name'])
            return HttpResponse("Hola this was succesful")
    else:
        form = UploadTrainImage()
    # return render(request, 'addTrainingImage.html', {'train_data_form':form})
    return render(request, 'base.html')

def trainImageHandler(img, class_name):
    with open(class_name+'/'+class_name+'.jpeg','wb+') as destination:
        for chunk in img.chunks():
            destination.write(chunk)


def testImage():
    pass




