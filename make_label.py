import cv2
import os
import random

def txt_make(path,save_path):
    listfiles = os.listdir(path) 
    if '.ipynb_checkpoints' in listfiles:
        listfiles.remove('.ipynb_checkpoints')
    for i,j in enumerate(listfiles):
        for t_t in ('train','test'):
            view_s_path = save_path+'/'+t_t+'/'+j
            isExists = os.path.exists(view_s_path)
            if not isExists:
                os.makedirs(view_s_path) 
                print(view_s_path+' 文件夹创建成功')
            view_files = path+'/'+j+'/'+t_t
            view_lists = os.listdir(view_files)
            if '.ipynb_checkpoints' in view_lists:
                view_lists.remove('.ipynb_checkpoints')
            txt_files = os.listdir(view_s_path)
            view_n = []
            for index,view_list in enumerate(view_lists):
                view_name = view_list[:-8]
                view_n.append(view_name)
            num_ = []
            txt_num = 0
            while len(view_n) != 0:
                view_name = view_n[0]
                num_ = [index for index,view_num in enumerate(view_n) if view_num == view_name]
                txt_path = view_s_path+'/'+view_name+'.txt'
                with open(txt_path,'w') as f:
                    f.write(str(i)+'\n')
                    f.write(str(len(num_))+'\n')
                txt_ = open (txt_path,'a+')
                sub_ = 0
                for num in num_:
                    view_ = view_lists[num-sub_]
                    txt_.write(view_files+'/'+view_+'\n')
                    view_n.remove(view_name)
                    view_lists.remove(view_)
                    sub_ += 1
                txt_.close()
                txt_num += 1
            print(view_s_path+'里的txt文件制作成功','txt文件数量：',txt_num)
    print('txt创建成功')

def txt_write(file_path,txt_files,s_p,t_t):
    if '.ipynb_checkpoints' in txt_files:
        txt_files.remove('.ipynb_checkpoints') 
    for i,j in enumerate(txt_files):
        txt_f_path = file_path+'/'+j
        txt_lists = os.listdir(txt_f_path)
        if '.ipynb_checkpoints' in txt_lists:
            txt_lists.remove('.ipynb_checkpoints')

        if t_t =='train':
            with open(s_p+'train_lists.txt','a+') as f:
                for txtlist in txt_lists:
                    if txtlist != '.ipynb_checkpoints':
                        txt_path = txt_f_path+'/'+txtlist
                        f.write(txt_path+' '+str(i)+'\n')
        else:
            num = len(txt_lists)
            random.shuffle(txt_lists)
            with open(s_p+'val_lists.txt','a+') as f:
                for txt_num in range(0,int(num/2)):
                    txt_path = txt_f_path+'/'+txt_lists[txt_num]
                    f.write(txt_path+' '+str(i)+'\n')
            with open(s_p+'test_lists.txt','a+') as f:
                for txt_num in range(int(num/2),num):
                    txt_path = txt_f_path+'/'+txt_lists[txt_num]
                    f.write(txt_path+' '+str(i)+'\n')