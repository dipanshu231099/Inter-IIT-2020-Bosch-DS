import os

p="/home/abhishek/django_project4/User_Custom_Train"
l=sorted(os.listdir(p))
n_classes=len(l)
Classes=l
new_classes=Classes[43:n_classes]

for i in new_classes:
    #path = "D:/Bosch/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/{0}/".format(num(i))
    path = "/home/abhishek/django_project4/train/{}".format(i)
    print(path)
