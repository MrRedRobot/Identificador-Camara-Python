# -*- coding: utf-8 -*-
"""
"""
from cv2 import *
import numpy as np
import os
import re
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

sport_model = Sequential()

def red():
    
    dirname = os.path.join(os.getcwd(), 'imagenes')
    imgpath = dirname + os.sep 
    
    images = []
    directories = []
    dircount = []
    prevRoot=''
    cant=0
    
    print("leyendo imagenes de ",imgpath)
    
    
    for root, dirnames, filenames in os.walk(imgpath):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                cant=cant+1
                filepath = os.path.join(root, filename)
                image = plt.imread(filepath)/255.0
                images.append(image)
                b = "Leyendo..." + str(cant)
                print (b, end="\r")
                if prevRoot !=root:
                    print(root, cant)
                    prevRoot=root
                    directories.append(root)
                    dircount.append(cant)
                    cant=0
    dircount.append(cant)
    
    dircount = dircount[1:]
    dircount[0]=dircount[0]+1
    print('Directorios leidos:',len(directories))
    print("Imagenes en cada directorio", dircount)
    print('suma Total de imagenes en subdirs:',sum(dircount))
    
    
    
    labels=[]
    indice=0
    for cantidad in dircount:
        for i in range(cantidad):
            labels.append(indice)
        indice=indice+1
    print("Cantidad etiquetas creadas: ",len(labels))
    
    señales=[]
    indice=0
    for directorio in directories:
        name = directorio.split(os.sep)
        print(indice , name[len(name)-1])
        señales.append(name[len(name)-1])
        indice=indice+1
    
    y = np.array(labels)
    X = np.array(images) #convierto de lista a numpy
    
    
    # Find the unique numbers from the train labels
    classes = np.unique(y)
    nClasses = len(classes)
    print('Total number of outputs : ', nClasses)
    print('Output classes : ', classes)
    
    #Mezclar todo y crear los grupos de entrenamiento y testing
    train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
    print('Training data shape : ', train_X.shape, train_Y.shape)
    print('Testing data shape : ', test_X.shape, test_Y.shape)
    
    
    #train_X = train_X.astype('float32')
    #test_X = test_X.astype('float32')
    
    
    # Change the labels from categorical to one-hot encoding
    train_Y_one_hot = to_categorical(train_Y)
    test_Y_one_hot = to_categorical(test_Y)    
    # Display the change for category label using one-hot encoding
    print('Original label:', train_Y[0])
    print('After conversion to one-hot:', train_Y_one_hot[0])
    
    train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
    
    print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)
    
    INIT_LR = 1e-3
    epochs = 6
    batch_size = 64
    
   
    sport_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(300,300,3)))
    sport_model.add(LeakyReLU(alpha=0.1))
    sport_model.add(MaxPooling2D((2, 2),padding='same'))
    sport_model.add(Dropout(0.5))
    
    sport_model.add(Flatten())
    sport_model.add(Dense(32, activation='linear'))
    sport_model.add(LeakyReLU(alpha=0.1))
    sport_model.add(Dropout(0.5)) 
    sport_model.add(Dense(nClasses, activation='softmax'))
    
    sport_model.summary()
    
    sport_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adagrad(lr=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])
    
    sport_train_dropout = sport_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
    
    test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)
    
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    """
    image = plt.imread('C:\\Users\\Edward Tejedor\\Desktop\\Proyecto final\\imagenes\\Prohibido parquear\\ProhibParq1.jpg')/255
    image = plt.imread('C:\\Users\\Edward Tejedor\\Desktop\\Proyecto final\\imagenes\\pare\\pare1.jpg')/255
    prev = []
    prev.append(image)
    
    prediccion(np.array(prev))
    
    
    
    """
    


def prediccion(image):
    prediccion = sport_model.predict(image)
    prediccion = sport_model.predict(np.array(prev))
    if prediccion[0][0] == 1:
        print("Pare")
    else:
        if prediccion[0][1] == 1:
            print("Prohibido")    


if __name__ == "__main__":
    
    red()
    
    namedWindow("webcam")
    cv = VideoCapture(0);
    """
    leido, frame = cv.read()
    frame = cv2.resize(frame, (300,300))
    if leido == True:
        cv2.imwrite("foto.png", frame)
        print("foto")
        prev = []
        prev.append(frame)
        a = np.array(prev)
        prediccion(a)
    else:
        print("Error con camara")
    cv.release()
    """
    while True:
        next, frame = cv.read()
        imshow("webcam", frame)
        """gray = cvtColor(frame, COLOR_BGR2GRAY)
        gauss = GaussianBlur(gray, (7,7), 1.5, 1.5)
        can = Canny(gauss, 0, 30, 3)
        cv2.imshow("filtro", can)
        """
        prev = []
        prev.append(cv2.resize(frame, (300,300)))
        a = np.array(prev)
        prediccion(a)
        
        if waitKey(50) >= 0:
            destroyAllWindows()
            break;