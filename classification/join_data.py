def num(i):
    i = str(i)
    return '0'*(5-len(i))+i

def combine(augs):
    data=[]
    labels=[]

    height = 32
    width = 32
    channels = 3
    classes = 43
    n_inputs = height * width*channels

    augs = ['horizontal_shift','brightness','zoom','horizontal_flip']
    n_classes=len(os.listdir("/home/abhishek/django_project4/train"))
    Classes=range(0,n_classes)

    for i in classes:
        #path = "D:/Bosch/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/{0}/".format(num(i))
        path = "/home/abhishek/django_project4/train"
        #print(path)
        if os.path.exists(path):
          Class=os.listdir(path)
          for a in Class:
            try:
                image=cv2.imread(path+a)
                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((height, width))
                data.append(np.array(size_image))
                labels.append(i)
                for augment in augs:
                    img=getattr(ags, augment)(np.array(size_image))
                    data.append(np.array(img))
                    labels.append(i)
            except AttributeError:
                print('N')

    np.save('/home/abhishek/django_project4/classification/model/new_data.npy', data) # save
    np.save('/home/abhishek/django_project4/classification/model/new_labels.npy', labels) # save
