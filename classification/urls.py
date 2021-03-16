from django.urls import path

from . import views

urlpatterns = [
    path('', views.addTrainingImage, name='index'),
    path('graphs/', views.graphs, name='plotly_graghs'),
    path('augment/', views.augment, name='augmentInput'),
]
