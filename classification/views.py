from django.shortcuts import render
from django.http import HttpResponse
import os
from .forms import UploadTrainImage
from plotly.offline import plot
from plotly.graph_objs import Scatter


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


def index(request):
    x_data = [0,1,2,3]
    y_data = [x**2 for x in x_data]
    plot_div = plot([Scatter(x=x_data, y=y_data,
                        mode='lines', name='test',
                        opacity=0.8, marker_color='green')],
               output_type='div')
    return render(request, "index.html", context={'plot_div': plot_div})
