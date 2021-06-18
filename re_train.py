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
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(dataset_train,dataset_val):
    print ('train() called')
    V = g_.NUM_VIEWS
    save_path = g_.MODEL_PATH
    batch__size = g_.BATCH_SIZE

    dataset_train.shuffle()
    dataset_val.shuffle()
    data_size = dataset_train.size()
    print ('training size:', data_size)
    model = re_model.MVCNN_model((224, 224, 3),V,g_.NUM_CLASSES)
    model.optimizer = keras.optimizers.Adam(lr=g_.INIT_LEARNING_RATE,decay=0.1)
    model.loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #     model.summary()

    steps = 0
    save_acc = 0
    all_val_loss = []
    all_val_acc = []
    all_epo_loss = []
    all_epo_acc = []
    for i in range(40):
        print('epoch:',i)
        all_pred = np.array([])
        all_losses = []
        all_y =[]
        for x,y in dataset_train._batches(batch__size):
            x = tf.transpose(x, perm=[1, 0, 2, 3, 4])
            x = tf.cast(x,dtype=tf.float32)
            t_x = []
            for j in range(V):
                t_x.append(tf.gather(x, j))
            with tf.GradientTape() as tape:
                pred = model(t_x)
                loss = model.loss_func(y,pred)
            
            gradients = tape.gradient(loss,model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            
            steps += 1
            pred = tf.nn.softmax(pred)
            pred = tf.argmax(pred,1)
            all_losses.append(loss)
            all_pred = np.hstack((all_pred, pred))
            all_y.extend(y)
            acc = metrics.accuracy_score(y,np.array(pred))
            print('loss=%.4f , acc=%.4f'%(loss,acc*100))
            # validation
            if steps %g_.VAL_PERIOD == 0:
                val_losses = []
                predictions = np.array([])
                val_y = []
                for val_batch_x,val_batch_y in dataset_val._batches(batch__size):
                    val_batch_x = tf.transpose(val_batch_x, perm=[1, 0, 2, 3, 4])
                    val_batch_x = tf.cast(val_batch_x,dtype=tf.float32)
                    v_x = []
                    for j in range(V):
                        v_x.append(tf.gather(val_batch_x, j))
                    pred = model(v_x)
                    softmax = tf.nn.softmax(pred)
                    val_pred = tf.argmax(softmax,1)
                    val_loss = model.loss_func(val_batch_y,pred)
                    val_losses.append(val_loss)
                    predictions = np.hstack((predictions, val_pred))
                    val_y.extend(val_batch_y)
                    
                val_loss = np.mean(val_losses)
                all_val_loss.append(val_loss)
                acc = metrics.accuracy_score(val_y[:predictions.size], np.array(predictions))
                all_val_acc.append(acc)
                print ('%s: step %d, val loss=%.4f, val acc=%f' %(datetime.now(), steps, val_loss, acc*100.))
                #保存网络
                if save_acc==0:
                    save_acc = acc
                else:
                    if save_acc<acc:
                        save_acc =acc
                        model.save(save_path)
                        print('model已保存，val_acc=%.4f'%save_acc)

        mean_loss = np.mean(all_losses)
        all_epo_loss.append(mean_loss)
        mean_acc = metrics.accuracy_score(all_y[:all_pred.size],np.array(all_pred))
        all_epo_acc.append(mean_acc)
        print('epoch %d: mean_loss=%.4f, mean_acc=%.4f'%(i,mean_loss,mean_acc*100))
    # loss、acc曲线图绘制
    x = np.arange(1,80,1)
    y = np.array(all_val_acc)
    plt.plot(x,y,label="val_acc")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("acc")
    plt.show()
    
    x = np.arange(1,41,1)
    y = np.array(all_epo_acc)
    plt.plot(x,y,label="epo_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()


def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels


def data_load():
    st = time.time() 
    print ('start loading data')
    listfiles_train, labels_train = read_lists(g_.TRAIN_LOL)
    listfiles_val,labels_val = read_lists(g_.VAL_LOL)
    dataset_train = Dataset(listfiles_train, labels_train, subtract_mean=False, V=g_.NUM_VIEWS)
    dataset_val = Dataset(listfiles_val, labels_val, subtract_mean=False, V=g_.NUM_VIEWS)
    print ('done loading data, time=', time.time() - st)
    return dataset_train,dataset_val