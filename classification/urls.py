from django.urls import path
from django.shortcuts import redirect
from . import views

urlpatterns = [
    path('',lambda request: redirect('AddTrainImage/', permanent=False) ),
    path('AddTrainImage/', views.addTrainingImage, name='addTrainingImage'),

    path('augment/', views.augment, name='augmentInput'),
    path('retrain/', views.re_train_model, name='retrainingdata'),
    path('retrain2/', views.direct , name="hub"),
    path('Merge/', views.Merge , name="hub2")
]
#path('graphs/', views.graphs, name='plotly_graghs'),
