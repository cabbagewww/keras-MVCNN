import cv2
import random
import numpy as np
import time
import globals as g_

W = H = 256

class Shape:
    def __init__(self, list_file):
        with open(list_file) as f:
            self.label = int(f.readline())
            self.V = int(f.readline())
            view_files = [l.strip() for l in f.readlines()]

        self.views = self._load_views(view_files, self.V)
        self.done_mean = False


    def _load_views(self, view_files, V):
        views = []
        for f in view_files:
            im = cv2.imread(f)
            im = cv2.resize(im, (W, H))
            # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) #BGR!!
            assert im.shape == (W,H,3), 'BGR!'
            im = im.astype('float32')
            views.append(im)
        views = np.asarray(views)
        return views

    def subtract_mean(self):
        if not self.done_mean:
            mean_bgr = (104., 116., 122.)
            for i in range(3):
                self.views[:,:,:,i] -= mean_bgr[i]

            self.done_mean = True

    def crop_center(self, size=(224,224)):  #AlexNet为227×227 ，VGG-M、ResNet为224×224
        w, h = self.views.shape[1], self.views.shape[2]
        wn, hn = size
        left = w / 2 - wn / 2
        top = h / 2 - hn / 2
        right = left + wn
        bottom = top + hn
        self.views = self.views[:, int(left):int(right), int(top):int(bottom), :]


class Dataset:
    def __init__(self, listfiles, labels, subtract_mean, V):
        self.listfiles = listfiles
        self.labels = labels
        self.shuffled = False
        self.subtract_mean = subtract_mean
        self.V = V
        print ('dataset inited')
        print ('  total size:', len(listfiles))

    def shuffle(self):
        z = list(zip(self.listfiles, self.labels))
        random.shuffle(z)
        self.listfiles, self.labels = [list(l) for l in zip(*z)]
        self.shuffled = True

    def _batches(self,batch_size):
        listfiles = self.listfiles
        n = len(listfiles)
        for i in range(0, n, batch_size):
            starttime = time.time()
            lists = listfiles[i : i+batch_size]
            #VGG-M、ResNet
            x = np.zeros((batch_size, self.V, 224, 224, 3))
            #AlexNet
#             x = np.zeros((batch_size, self.V, 227, 227, 3))
            y = np.zeros(batch_size)

            for j,l in enumerate(lists):
                s = Shape(l)
                s.crop_center()
                if self.subtract_mean:
                    s.subtract_mean()
                x[j, ...] = s.views
                y[j] = s.label
            
#             print ('load batch time:', time.time()-starttime, 'sec')
            yield x, y
    

    def size(self):
        """ size of listfiles (if splitted, only count 'train', not 'val')"""
        return len(self.listfiles)