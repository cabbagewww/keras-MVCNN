import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
from tensorflow import keras
from input import Dataset
import globals as g_
import re_model
import sklearn.metrics as metrics
from tensorflow.keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels


def data_load():
    st = time.time() 
    print ('start loading data')
    listfiles_test,labels_test = read_lists(g_.TEST_LOL)
    dataset_test = Dataset(listfiles_test, labels_test, subtract_mean=False, V=g_.NUM_VIEWS)
    print ('done loading data, time=', time.time() - st)
    return dataset_test

def test(dataset_test):
    print ('test() called')
    V = g_.NUM_VIEWS
    save_path = g_.MODEL_PATH
    batch__size = g_.BATCH_SIZE

    dataset_test.shuffle()
    data_size = dataset_test.size()
    print ('testing size:', data_size)
#     model = re_model.MVCNN_model((227, 227, 3),V,g_.NUM_CLASSES)
    model = load_model(save_path, custom_objects={'tf': tf},compile=False)
    model.optimizer = keras.optimizers.Adam(lr=g_.INIT_LEARNING_RATE,decay=0.1)
    model.loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    steps = 0
    save_acc = 0
    predictions = np.array([])
    test_losses = []
    test_y =[]
    for batch_x,batch_y in dataset_test._batches(batch__size):
        x = tf.transpose(batch_x, perm=[1, 0, 2, 3, 4])
        x = tf.cast(x,dtype=tf.float32)
        t_x = []
        for j in range(V):
            t_x.append(tf.gather(x, j))
        pred = model(t_x)
        softmax = tf.nn.softmax(pred)
        test_pred = tf.argmax(softmax,1)
        test_loss = model.loss_func(batch_y,pred)
        test_losses.append(test_loss)
        predictions = np.hstack((predictions, test_pred))
        test_y.extend(batch_y)
        test_acc = metrics.accuracy_score(batch_y,np.array(test_pred))
        print('loss = %.4f,acc = %.4f'%(test_loss,test_acc))
    test_loss = np.mean(test_losses)
    acc = metrics.accuracy_score(test_y[:predictions.size], np.array(predictions))
    print ('%s:  test loss=%.4f, test acc=%f' %(datetime.now(),  test_loss, acc*100.))