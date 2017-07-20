import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import xml.etree.ElementTree as ET
from scipy.misc import imread
import random
import cv2

############################################
########### DATA shuffle
############################################

root_folder = "/home/federico/NAS/HumanRecording/"

#shuffle the data
list_person = root_folder+"DATApython/list_persons.npy"
list_persons=np.load(list_person)
print("data shuffling...")
test_data=[]
test_labls=[]
for person in range(len(list_persons)):
     person_name = (list_persons[person]).split("/")[::-1][0]
     print(person_name)
     for session in range(0,5):
        print("session="+str(session))
        filename_dat = root_folder + "DATApython/DATAformatedForTraining2/cam_3_train_data_"+str(person_name)+str(session)+".npy"
        filename_lab = root_folder + "DATApython/DATAformatedForTraining2/cam_3_train_labels_"+str(person_name)+str(session)+".npy"
        train_data_shuf = np.load(filename_dat)
        train_labels_shuf = np.load(filename_lab)
        n_samples = np.shape(train_labels_shuf)[0]
        #index_shuffle = random.sample(range(0, n_samples), n_samples)
        #index_shuffle = np.reshape(index_shuffle,(len(index_shuffle)))
        #train_data_shuf = [train_data_shuf[i] for i in index_shuffle]
        #train_labels_shuf = [train_labels_shuf[i] for i in index_shuffle]
        
        #np.save("/home/inilabs/NAS/HumanRecording/DATApython/DATAformatedForTraining2/cam_3_train_data_"+str(person_name)+str(session)+".npy", train_data_shuf[0:int(0.95*len(train_data_shuf))])
        #np.save("/home/inilabs/NAS/HumanRecording/DATApython/DATAformatedForTraining2/cam_3_train_labels_"+str(person_name)+str(session)+".npy", train_labels_shuf[0:int(0.95*len(train_labels_shuf))])

        #print(n_samples)
        #print(int(0.95*len(train_data_shuf)))

        if(test_data==[]):
           test_data=train_data_shuf[int(0.1*len(train_data_shuf))-1:int(0.2*len(train_data_shuf))]
           test_labls=train_labels_shuf[int(0.1*len(train_labels_shuf))-1:int(0.2*len(train_labels_shuf))]
        else:
           test_data=np.append(test_data,train_data_shuf[int(0.1*len(train_data_shuf))-1:int(0.2*len(train_data_shuf))], axis=0)
           test_labls=np.append(test_labls,train_labels_shuf[int(0.1*len(train_labels_shuf))-1:int(0.2*len(train_labels_shuf))], axis=0)
np.save(root_folder + "DATApython/DATAformatedForTraining2/cam_3_test_data_3.npy", test_data)
np.save(root_folder + "DATApython/DATAformatedForTraining2/cam_3_test_labels_3.npy", test_labls)
