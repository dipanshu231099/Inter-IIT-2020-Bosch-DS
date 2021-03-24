# 1 Installation and user manual
This document gives easy steps to install/set-up and to use the web app which will enable to user to interact with the model.

## 1.1 Installation
The user needs to have django installed togethor with the dependancies to run the model most latest dependnecies will work except there is a strict requirements for (python 3.7.10 and tensorflow 1.15).
The best way to do so is by cereating a python virtual environment.
### 1.1.1 Creating a virtual env with python 3.7.10 version
 >  To install the virtual environ ment run the following code
 ```
$ python3.7 -m pip install --user virtualenv
 ```

 >To create a virtual environment<br>
 - Go to the directory where you want to create the virtual env and run the following command in the terminal.

 
 ```
$ python3.7 env -m env-name
 ```
 - Then activate the virtual env using the following command.

 ```
 $ source env-name/bin/activate
 ```
 
>Loading the necessary Packages<br>
- In order to install all the necessary packages in the virtual env cd into the project directory.
- There will be a requirements.txt this file contains the name of all the packages that are being used along with their versions.
- Now to install the pacages from this file run the following command.
```
$ pip install -r requirements.txt 
```

### 1.1.2 Run the Django server.

- cd into the project directory and run the following commands to start the server.
```
python3.7 manage.py runserver
```
- make migrations, the make migrations will only be needed to be done once.
```
$ python3.7 manage.py makemigrations

$ python3.7 manage.py migrate
```


## 1.2 Usage
After carefully going over the problem statement the functionalty have been divided onto broadly four types/coditions.

1. The user should be able to enter images in the training set which consists of 42 default classes and 5 additional classes added by us.

2. The user should be able to retrain the model after making augmentations to the images in the newly added 5 classes and then merging the augmented image with the existing dataset to increase the difficulty OR  apply augmentations to the whole training set and then merge it with the exixting trainign set in both the cases the difficulty and the scale of the training data would be increased and see the statistics of the model.
3. Test the model for any image.
4. Make neuron level analysis so that the reason for the models performance can be found.
5. The user should be able to make augmentations to just the original 43 classes and then add the augmented images to the existing training set thereby increaseing the data set and its difficulty.
6. Display and compare the results of retaining with the initial original trained model and the new one through varoius graphs.

To solve the above cases diffrent pathways(url paths/links) have been made with diffrent functionalities.Below is the the page level explanation of the functionalities.

### 1.1.2 Homepage.

#### Navbar
Everypage will have the navbar and The navbar has two buttons:<br>
1. Home:<br>
This button will bring you back to the home page from anywhere
2. Add images:<br>
This will take you to "/app/AddTrainImage/" . The form will will enable the user to entern n number of images to any one of the 48 classes.
3. Test Images:<br>
This will take you to "/app/TestImage/" . The form will will enable the user to enter a image and predict its class.


#### Localhost:8000/app/home:

The page has 4 pipelines all adressing one of the above problems.<br>
- The first button will simply retrain the original model of 43 classes and show that the default graphs that are displayed later in the webpage will match the new graphs that would be generated.<br>
>button1-->retraining-->graphs

- The second button will take you to the augmenttation page and enable you to add augmentations to the original training data then the user will be displayed one randomply chosen pair or original vs augmented from each class.After verifiying that the augmentations are corrctly done the user can then retrain the data after which he will be displayed graphs of the model before and after augmentations.<br>
>button2-->augmentation-->original vs augmented images-->retraining-->graphs

- The third button will take you to the add images page where the user can add images to the 48 classes and then he would be directed to an intermediate page that would check if the user wants to apply the augmentations to the newly added 5  classes or to the whole data set.After which the user would be directed to a webpage where he can see the original vs augmented image from each augmented class.The augmeted images would then be added with the training data and the model would be trained again.The page will then be directed to graphs webpage where he can see the graphs showing the statistics of the original vs the augmented model .<br>
>button2-->augmentation-->original vs augmented images-->retraining-->graphs






