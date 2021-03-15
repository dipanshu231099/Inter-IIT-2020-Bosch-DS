from django import forms
from .models import *

class UploadTrainImage(forms.Form):
    choices = [
        ('class1','class1'),
        ('class2','class2'),
        ('class3','class3'),
        ('class4','class4'),
        ('class5','class5'),
    ]
    class_name = forms.ChoiceField(choices=choices)
    img_file = forms.FileField()