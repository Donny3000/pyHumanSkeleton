# 2017, sophie.skriabine@gmail.com
# data preparation and tensor flow classifications

import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import xml.etree.ElementTree as ET
from scipy.misc import imread
import random
import cv2

############################################
########### DATA FOR TRAINING
############################################
root_folder = "/home/federico/NAS/HumanRecording/"

def release_list(a):
  del a[:]
  del a

list_person = root_folder + "DATApython/list_persons.npy"
list_persons=np.load(list_person)


print("start creating vectors...")
for counter in range(0,100):
  train_data_final = []
  train_labels_final = []
  np.save(root_folder + "DATApython/DATAformatedForTraining/cam_3_train_data_"+str(counter)+".npy", train_data_final)
  np.save(root_folder + "DATApython/DATAformatedForTraining/cam_3_train_labels_"+str(counter)+".npy", train_labels_final)
  #release_list(train_data_final)
  #release_list(train_labels_final)
print("done")

for person in range(len(list_persons)):
     person_name = (list_persons[person]).split("/")[::-1][0]
     print(person_name)
     for session in range(1,6):
      #if ((person>1)|(session> 1)):
          #release_list(train_data)
          #release_list(train_labels)
      # Load training and eval data
      print(session)
      try:
        train_filename_data = root_folder + "HumanRecording/DATApython/cam_3_train_X"+"_"+str(person_name)+"_"+"session"+str(session)+".npy"
        #print(train_filename_data)
        train_filename_labels = root_folder + "HumanRecording/DATApython/movements_id_train_Y"+"_"+str(person_name)+"_"+"session"+str(session)+".npy"
        
        #test_filename_data = "/home/inilabs/NAS/HumanRecording/DATApython/cam_3_test_X"+"_"+str(person_name)+"_"+"session"+str(session)+".npy"
        #test_filename_labels = "/home/inilabs/NAS/HumanRecording/DATApython/movements_id_test_Y"+"_"+str(person_name)+"_"+"session"+str(session)+".npy"
        scale_input_image_size_x = 64
        scale_input_image_size_y = 64
        camera_size_y = 261
        camera_size_x = 346
    
        train_data = np.load(train_filename_data)
        train_labels = np.load(train_filename_labels)
        #train_data = read_npy_chunk(train_filename_data,int(0+counter*0.01*len(train_data)),int((counter+1)*0.01*len(train_data))
        #train_labels = read_npy_chunk(train_filename_labels, int(0+counter*0.01*len(train_labels)),int((counter+1)*0.01*len(train_labels))
      except:
        print("file nor found")
        #print(train_data.size())
        #print(train_labels.size())
      for counter in range(0,100):
        print("counter="+str(counter))
        print(int(0+counter*0.01*len(train_data)))
        print(int((counter+1)*0.01*len(train_data)))
        for index in range(int(0+counter*0.01*len(train_data)),int((counter+1)*0.01*len(train_data))):
          filename_data = root_folder + "DATApython/DATAformatedForTraining/cam_3_train_data_"+str(counter)+".npy"
          filename_labels = root_folder + "DATApython/DATAformatedForTraining/cam_3_test_labels_"+str(counter)+".npy"
          train_data_final = np.load(train_filename_data)
          train_labels_final = np.load(train_filename_labels)

          np.append(train_data_final,train_data[index])
          np.append(train_labels_final,train_labels[index])
          #print(train_data_final)
          #print(train_labels_final)
          np.save(root_folder + "DATApython/DATAformatedForTraining/cam_3_train_data_"+str(counter)+".npy", train_data_final)
          np.save(root_folder + "DATApython/DATAformatedForTraining/cam_3_test_labels_"+str(counter)+".npy", train_labels_final)
