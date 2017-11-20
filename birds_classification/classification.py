from os.path import basename, join
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, History
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from os.path import basename, join
from glob import glob
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D

from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.utils import print_summary
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop


def train_classifier(train_gt, train_img_dir, fast_train = True):    
    new_height = 200
    new_width = 200
    num_classes = 50
    
    jpeg_list = sorted(glob(join(train_img_dir, '*jpg')))
    num_samples = len(jpeg_list)
    
    X = np.zeros([num_samples, new_height, new_width, 3])
    y = np.zeros([num_samples, num_classes])
    yy = np.zeros([num_samples])
    
    count = 0
    for path in jpeg_list:
        image = imread(path)
        
        # resize the image
        scale_width = image.shape[1] / new_width
        scale_height = image.shape[0] / new_height
        image_resized = resize(image, (new_height, new_width))
        
        # handle grayscale images
        if len(image_resized.shape) == 2:
            tmp_image = np.zeros([new_height, new_width, 3])
            tmp_image[:, :, 0] = image_resized
            tmp_image[:, :, 1] = image_resized
            tmp_image[:, :, 2] = image_resized
            image_resized = tmp_image
        
        X[count] = image_resized
        class_ind = int(train_gt[basename(path)])
        y[count, class_ind] = 1
        yy[count] = int(class_ind)
        
        count = count + 1
    
    X_train, X_test, yy_train, yy_test = train_test_split(X, yy, test_size = 0.2,
                                                        random_state = 2017, shuffle = True,
                                                        stratify = yy)
        
    y_train = np.zeros([len(yy_train), num_classes])
    
    for i in range(len(yy_train)):
        class_ind = int(yy_train[i]) 
        y_train[i, class_ind] = 1

    y_test = np.zeros([len(yy_test), num_classes])        
    for i in range(len(yy_test)):
        class_ind = int(yy_test[i])
        y_test[i, class_ind] = 1
    
    base_model = ResNet50(include_top = False, weights='imagenet', input_shape = [new_height, new_width, 3])
    
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    num_layers = len(base_model.layers)
    ind_layer = 1
    for layer in base_model.layers:
        if ind_layer < (num_layers - 7): # do not froze the last five layers
            layer.trainable = False
        ind_layer = ind_layer + 1
    
    # compile the model
    rmps_prop = RMSprop(lr=0.00001)
    model.compile(optimizer = rmps_prop, loss = 'categorical_crossentropy', metrics=['accuracy'])

    epochs = 4000
    batch_size = 50

    if fast_train is True:
        epochs = 1
    
    model.fit(X_train, y_train,
              batch_size=batch_size,
              validation_data = (X_test, y_test),
              epochs=epochs
             )

    pass


def classify(model, test_img_dir):
    new_height = 200
    new_width = 200
    num_classes = 50
    
    jpeg_list = sorted(glob(join(test_img_dir, '*jpg')))
    num_samples = len(jpeg_list)
    
    X = np.zeros([num_samples, new_height, new_width, 3])
    y = np.zeros([num_samples, num_classes])
    
    count = 0
    test_gt = {}
    test_filenames = []
    for path in jpeg_list:
        image = imread(path)
        test_gt[basename(path)] = 0
        
        # resize the image
        scale_width = image.shape[1] / new_width
        scale_height = image.shape[0] / new_height
        image_resized = resize(image, (new_height, new_width))
        
        # handle grayscale images
        if len(image_resized.shape) == 2:
            tmp_image = np.zeros([new_height, new_width, 3])
            tmp_image[:, :, 0] = image_resized
            tmp_image[:, :, 1] = image_resized
            tmp_image[:, :, 2] = image_resized
            image_resized = tmp_image
        
        X[count] = image_resized
        test_filenames.append(basename(path))
        count = count + 1
    
    yy = model.predict(X)
    
    for picture_ind in range(len(test_filenames)):
        class_ind = np.argmax(yy[picture_ind])
        test_filename = test_filenames[picture_ind]
        test_gt[test_filename] = class_ind

    return test_gt
