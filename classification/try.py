import os
base_dir = os.getcwd()

<<<<<<< HEAD
# p="/home/abhishek/django_project4/User_Custom_Train"
# l=sorted(os.listdir(p))
# n_classes=len(l)
# Classes=l
# new_classes=Classes[0:n_classes]


a="rotaiton,horizontal,blur"
print(a.split(","))
=======
p=base_dir+"/User_Custom_Train"
l=sorted(os.listdir(p))
n_classes=len(l)
Classes=l
new_classes=Classes[43:n_classes]

for i in new_classes:
    #path = "D:/Bosch/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/{0}/".format(num(i))
    path = base_dir+"/train/{}".format(i)
    print(path)
>>>>>>> fec6aad0510eda69ee64e2bade554a3f21567543
