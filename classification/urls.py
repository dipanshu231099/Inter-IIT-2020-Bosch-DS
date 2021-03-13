from django.urls import path

from . import views

urlpatterns = [
    path('', views.addTrainingImage, name='index'),
]
