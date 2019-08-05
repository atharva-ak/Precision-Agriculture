### Gets external test data for extra verification #### 

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
# from biosppy.signals import ecg
import cv2
from scipy import signal, arange,stats
import numpy.fft as fft
import matplotlib.pylab as plt
import glob
import os
import warnings
import sys
import pandas as pd
import numpy as np
import os
from glob import glob
import random
import matplotlib.gridspec as gridspec
import seaborn as sns
import zlib
import itertools
import cv2
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.mobilenet import MobileNet

# classLabel={'img_NSR':0, 'img_AFIB':1,'img_APB':2, 'img_LBBBB':3, 'img_RBBBB':4, 'img_PVC':5, 'img_PR':6}
classLabel={'defected':0, 'normal':1}

trainDir = './binary_img/'
trainImage = []
trainLabels = []

external_testDir = './binary_test/'

x = [] # images as arrays
y = [] # labels
WIDTH = 221
HEIGHT = 221

kernel = np.ones((2,2),np.uint8)

for root,dirs,file in os.walk(trainDir):
    for name in file:
        fileName =os.path.join(root,name)
        full_size_image = cv2.imread(fileName)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        className = root.split("/")[-1]
        mylabel = classLabel[className]
        y.append(mylabel)

ext_X = []
ext_Y = []    
for root,dirs,file in os.walk(external_testDir):
    for name in file:
        fileName =os.path.join(root,name)
        full_size_image = cv2.imread(fileName)
        ext_X.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        className = root.split("/")[-1]
        mylabel = classLabel[className]
        ext_Y.append(mylabel)
        
        
X=np.array(x)
# print(X.shape)

EXT_X=np.array(ext_X)
print("External Test Data Shape:",EXT_X.shape)
EXT_Y = np.array(ext_Y)
Ext_Y_testHot = to_categorical(EXT_Y)
print("External Test Data Label Shape:",EXT_Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, X_test.shape)

X_train = X_train 
Y_train = np.array(Y_train)
X_test = X_test 
Y_test = np.array(Y_test)

print("Training Data Shape:", Y_train.shape)
print("Testing Data Shape:", X_test.shape)
print("Training Data Shape:", len(X_train), X_train[0].shape)
print("Testing Data Shape:", len(X_test), X_test[0].shape)

Y_trainHot = to_categorical(Y_train)
Y_testHot = to_categorical(Y_test)

from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced', np.unique(y), y)
print(class_weight)

# dict_characters = {0: 'img_NSR', 1: 'img_AFIB',2: 'img_APB', 3: 'img_LBBBB', 4: 'img_RBBBB', 5: 'img_PVC', 6: 'img_PR'}
dict_characters = {0: 'defected', 1: 'normal'}

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)

def plotKerasLearningCurve():
    plt.figure(figsize=(10,5))
    plt.title('Ext_learning_curve')
    metrics = np.load('./logs.npy',allow_pickle=True)[()]
    filt = ['acc'] # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')
    plt.savefig('./VGG_Ext_learning_curve.png')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
   
    plt.figure(figsize = (5,5))
    plt.tight_layout()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.yticks([])
    plt.xticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('CM ValUE',cm)
    thresh = cm.max() / 2.0
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "red")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./VGG_Ext_Conf_matrix.png', bbox_inches = "tight")
    
def plot_confusion_matrix_int(cm, classes,
                          normalize=False,
                          title='Confusion matrix_int',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
   
    plt.figure(figsize = (5,5))
    plt.tight_layout()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.yticks([])
    plt.xticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('CM ValUE',cm)
    thresh = cm.max() / 2.0
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "red")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./VGG_Ext_Conf_matrix_int.png', bbox_inches = "tight")

def plot_learning_curve(history):
    plt.figure(figsize=(50,10))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('./Ext_accuracy_curve.png')
    #plt.clf()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./VGG_Ext_loss_curve.png', bbox_inches = "tight")


from keras.applications.vgg16 import VGG16
from keras.models import Model
weight_path = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
im_size = 221
map_characters=dict_characters
def vgg16network(a,b,c,d,e,f,g,h,k):
    num_class = f
    epochs = g
    base_model = VGG16(include_top=False,weights='imagenet',input_shape=(im_size, im_size, 3))
    # Add a new top layer
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(num_class, activation='softmax')(x)
    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # First: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy', 
                  optimizer=keras.optimizers.RMSprop(lr=0.00001), 
                  metrics=['accuracy'])
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    model.summary()
#     model.fit(a,b, epochs=epochs, class_weight=e, validation_data=(c,d), verbose=1,callbacks = [MetricsCheckpoint('logs')])
    A = model.fit(a,b, epochs=epochs, class_weight=e, validation_data=(c,d), verbose=1,callbacks = [MetricsCheckpoint('logs')])
    print(model.metrics_names)
    score = model.evaluate(c,d, verbose=0)
    model.save('Binary_model.h5')
    print('\nKeras CNN #2 - accuracy:', score[1], '\n')
    
    y_pred_int = model.predict(c)
    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred_int, axis=1), target_names=list(map_characters.values())), sep='') 
    Y_pred_classes_int = np.argmax(y_pred_int,axis = 1) 
    Y_true_int = np.argmax(d,axis = 1) 
    confusion_mtx_int = confusion_matrix(Y_true_int, Y_pred_classes_int)
    plot_confusion_matrix_int(confusion_mtx_int, classes = list(map_characters.values()))
    
    
    y_pred = model.predict(h)
    print('\n', sklearn.metrics.classification_report(np.where(k > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='') 
    Y_pred_classes = np.argmax(y_pred,axis = 1) 
    Y_true = np.argmax(k,axis = 1) 
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values()))
#     print('Ytrue',Y_true)
#     print('Ypred',Y_pred_classes)
    plot_learning_curve(A)
    plotKerasLearningCurve()
#    plt.show()
    
#    plt.show()
    return model
vgg16network(X_train, Y_trainHot, X_test, Y_testHot,class_weight,2,100,EXT_X,Ext_Y_testHot) 
