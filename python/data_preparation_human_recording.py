# 2017, federico.corradi@inilabs.com
# data preparation and tensor flow classifications

import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import xml.etree.ElementTree as ET
from scipy.misc import imread
import random
import cv2

############################################
########### DATA LOADER
############################################

# define protocol 
#which_sessions_and_movement_name = {1:['leftarmabduction','rightarmabduction',3,4,5,6,7,8],2:[1,2,3,4,5,6],3:[1,2,3,4,5,6],4:[1,2,3,4,5,6],5:[1,2,3,4,5,6,7]}
#which_sessions_and_movements_id = {1:[1,2,3,4,5,6,7,8],2:[1,2,3,4,5,6],3:[1,2,3,4,5,6],4:[1,2,3,4,5,6],5:[1,2,3,4,5,6,7]} # session starts from 1

root_folder = "/home/federico/NAS/HumanRecording/"
recordings_folder = root_folder + "Database/DVSwithVicon/"
recordings_date = [f for f in listdir(recordings_folder) if isdir(join(recordings_folder, f))]
# get the avaiable people 
recordings_data_sessions = []
data_recordings = []
for i in range(len(recordings_date)):
    data_dir = recordings_folder + recordings_date[i] + "/Data/poseImages/"
    if(len([f for f in listdir(data_dir) if isdir(join(data_dir, f))]) > 0):
        recordings_data_sessions.append([f for f in listdir(data_dir) if isdir(join(data_dir, f))])
        data_recordings.append(recordings_date[i])

#recordings_person = [val for sublist in recordings_person for val in sublist] #flattern list

#final sessions data folder
sessions_folder = []
for person_id in range(len(recordings_data_sessions)):
    if(len(recordings_data_sessions[person_id]) > 1):
        for person_id_a in range(len(recordings_data_sessions[person_id])):
           sessions_data_folder = (recordings_folder + data_recordings[person_id] + "/Data/poseImages/" + str(recordings_data_sessions[person_id][person_id_a]).strip('[]\'')) 
           sessions_folder.append(sessions_data_folder)
    else:
        sessions_data_folder = (recordings_folder + data_recordings[person_id] + "/Data/poseImages/" + str(recordings_data_sessions[person_id]).strip('[]\''))
        sessions_folder.append(sessions_data_folder)

def release_list(a):
   del a[:]
   del a

print("save the list of the subject")
np.save("/home/inilabs/NAS/HumanRecording/DATApython/list_persons.npy",sessions_folder)

#get movements
for this_person in range(len(sessions_folder)):

    all_sessions_this_person = [f for f in listdir(sessions_folder[this_person]) if isdir(join(sessions_folder[this_person], f))]

    for this_session in range(len(all_sessions_this_person)):


        if((this_person > 1)|(this_session> 1)):
            release_list(mega_vector_folder)
            release_list(timestamps)
            release_list(labels)
            release_list(images)
            release_list(movements_id_db)
            release_list(cam_1)
            release_list(cam_2)
            release_list(cam_3)
            release_list(cam_4)
            release_list(all_cams)

        mega_vector_folder = []
        print("Looking into foder for data and organize them...")
        timestamps = []     #us timestamp
        labels = []         #vector coordinates
        images = []          #png images
        movements_id_db = []    #action id
        cam_1 = []
        cam_2 = []
        cam_3 = []
        cam_4 = []
        all_cams = []
   

    
        dir_movements = sessions_folder[this_person] + "/"+str(all_sessions_this_person[this_session])
        #print(dir_movements)
        movements_id = ([f for f in listdir(dir_movements) if isfile(join(dir_movements, f))])
        if(len(movements_id)%2 != 0):
            print("Avi/xml not labelled")
            raise Exception
        for this_mov in range(len(movements_id)): 
            if(movements_id[this_mov].endswith("avi")):
                #add movements id
                id_mov = int(str(all_sessions_this_person[this_session].strip('session'))+str(int(movements_id[this_mov].strip("mov").strip(".avi"))))
                #grub images
                video_file_name  = dir_movements + "/" + movements_id[this_mov]
                vidcap = cv2.VideoCapture(video_file_name)
                if(not vidcap):
                    print("vidcap is empty")
                    raise Exception
                cam_1_t = []
                cam_2_t = []
                cam_3_t = []
                cam_4_t = []
                all_cams_t = []
                movements_id_tmp = []
                success = True;
                count = 0;
                while success:
                    success,image = vidcap.read()
                    if(success == False):
                        cam_1.append(cam_1_t)
                        cam_2.append(cam_2_t)
                        cam_3.append(cam_3_t)
                        cam_4.append(cam_4_t)
                        all_cams.append(all_cams_t)
                        movements_id_db.append(movements_id_tmp)
                        break;
                    cam_1_t.append(image[:,0:346,1])
                    cam_2_t.append(image[:,346:346*2,1])
                    cam_3_t.append(image[:,346*2:346*3,1])
                    cam_4_t.append(image[:,346*3:346*4,1])
                    all_cams_t.append(image[:,:,1])
                    movements_id_tmp.append(id_mov)
                    count += 1
                
                #get same xml    
                name_file = movements_id[this_mov].strip('avi')+"xml"
                #grab frames   
                xml_file_name  = dir_movements + "/" + name_file
                labels.append(ET.parse(xml_file_name))

    	#flattern all lists
        cam_1 = [val for sublist in cam_1 for val in sublist]
        cam_2 = [val for sublist in cam_2 for val in sublist]
        cam_3 = [val for sublist in cam_3 for val in sublist]
        cam_4 = [val for sublist in cam_4 for val in sublist]
        all_cams = [val for sublist in all_cams for val in sublist]
        movements_id_db = [val for sublist in movements_id_db for val in sublist]
        labels_flat = []
        for sublist in labels:
            for x in sublist.findall("mov"):
                labels_flat.append(x)

	    #check files are not empty
	    if(not cam_1):
			 print(cam_1)
			 print("cam1 is empty")
			 raise Exception

	    # Training Data
	    n_samples = np.shape(cam_1)[0]
	    test_perc = 0.3
	    test_num = int(np.floor(n_samples * 0.3))
	    train_num = n_samples - test_num

	    index_shuffle = random.sample(range(0, n_samples), n_samples)
	    index_shuffle = np.reshape(index_shuffle,(len(index_shuffle)))

        #print(n_samples)
        #print(len(labels_flat))
        #print(index_shuffle)
	    #devide dataset in train and test
        cam_1_s = [cam_1[i] for i in index_shuffle]
        cam_2_s = [cam_2[i] for i in index_shuffle]
        cam_3_s = [cam_3[i] for i in index_shuffle]
        cam_4_s = [cam_4[i] for i in index_shuffle]
        labels_flat = [labels_flat[i] for i in index_shuffle]
        movements_id_db = [movements_id_db[i] for i in index_shuffle]
        all_cams = [all_cams[i] for i in index_shuffle]

        train_X = {'cam_1': cam_1_s[0:train_num], 'cam_2': cam_2_s[0:train_num], 'cam_3': cam_3_s[0:train_num], 'cam_4': cam_4_s[0:train_num]}
        train_Y = movements_id_db[0:train_num]

        test_X = {'cam_1': cam_1_s[train_num:train_num+test_num], 'cam_2': cam_2_s[train_num:train_num+test_num], 'cam_3': cam_3_s[train_num:train_num+test_num], 'cam_4': cam_4_s[train_num:train_num+test_num]}
        test_Y = movements_id_db[train_num:train_num+test_num]

	    #save processed data TRAIN
        print("Flattern Train Data..")
        cam_1_f = np.zeros([len(train_X['cam_1']), 261*346])
        cam_2_f = np.zeros([len(train_X['cam_2']), 261*346])
        cam_3_f = np.zeros([len(train_X['cam_3']), 261*346])
        cam_4_f = np.zeros([len(train_X['cam_4']), 261*346])
        all_cams_f = np.zeros([len(train_X['cam_4']), 261*346*4])
        for i in range(len(train_X['cam_1'])):
		    lcam1 = np.shape(train_X['cam_1'][i])[0]*np.shape(train_X['cam_1'][i])[1]
		    cam_1_f[i][0:lcam1]  = np.reshape(train_X['cam_1'][i], [lcam1])   
		    lcam2 = np.shape(train_X['cam_2'][i])[0]*np.shape(train_X['cam_2'][i])[1] 
		    cam_2_f[i][0:lcam2]  = np.reshape(train_X['cam_2'][i], [lcam2])
		    lcam3 = np.shape(train_X['cam_3'][i])[0]*np.shape(train_X['cam_3'][i])[1]
		    cam_3_f[i][0:lcam3]  = np.reshape(train_X['cam_3'][i], [lcam3])
		    lcam4 = np.shape(train_X['cam_4'][i])[0]*np.shape(train_X['cam_4'][i])[1]
		    cam_4_f[i][0:lcam4]  = np.reshape(train_X['cam_4'][i], [lcam4])
		    all_cams_f[i][0:np.shape(all_cams[i])[0]*346*4] = np.reshape(all_cams[i], [np.shape(all_cams[i])[0]*346*4] )

        print("Done.")

	    #save processed data TEST
        print("Flattern Test Data")
        cam_1_y = np.zeros([len(test_X['cam_1']), 261*346])
        cam_2_y = np.zeros([len(test_X['cam_2']), 261*346])
        cam_3_y = np.zeros([len(test_X['cam_3']), 261*346])
        cam_4_y = np.zeros([len(test_X['cam_4']), 261*346])
        all_cams_y = np.zeros([len(test_X['cam_4']), 261*346*4])
        for i in range(train_num,n_samples):
		    lcam1 = np.shape(test_X['cam_1'][i-train_num])[0]*np.shape(train_X['cam_1'][i-train_num])[1]
		    cam_1_y[i-train_num][0:lcam1]  = np.reshape(test_X['cam_1'][i-train_num], [lcam1])   
		    lcam2 = np.shape(test_X['cam_2'][i-train_num])[0]*np.shape(train_X['cam_2'][i-train_num])[1] 
		    cam_2_y[i-train_num][0:lcam2]  = np.reshape(test_X['cam_2'][i-train_num], [lcam2])
		    lcam3 = np.shape(test_X['cam_3'][i-train_num])[0]*np.shape(train_X['cam_3'][i-train_num])[1]
		    cam_3_y[i-train_num][0:lcam3]  = np.reshape(test_X['cam_3'][i-train_num], [lcam3])
		    lcam4 = np.shape(test_X['cam_4'][i-train_num])[0]*np.shape(train_X['cam_4'][i-train_num])[1]
		    cam_4_y[i-train_num][0:lcam4]  = np.reshape(test_X['cam_4'][i-train_num], [lcam4])
		    all_cams_y[i-train_num][0:np.shape(all_cams[i-train_num])[0]*346*4] = np.reshape(all_cams[i-train_num], [np.shape(all_cams[i-train_num])[0]*346*4] )

        print("Done.")
        movements_id_train_Y = movements_id_db[0:train_num]
        movements_id_test_Y = movements_id_db[train_num:n_samples]

        labels_train_Y = labels_flat[0:train_num]
        labels_test_Y = labels_flat[train_num:n_samples]

        #save processed data
        session_number=(all_sessions_this_person[this_session]).split("/")[::-1][0]
        person_name = (sessions_folder[this_person]).split("/")[::-1][0]
        np.save(root_folder + "DATApython/cam_1_train_X_"+str(person_name)+"_"+str(session_number)+".npy", cam_1_f)
        np.save(root_folder + "DATApython/cam_1_test_X_"+str(person_name)+"_"+str(session_number)+".npy", cam_1_y)

        np.save(root_folder + "DATApython/cam_2_train_X_"+str(person_name)+"_"+str(session_number)+".npy", cam_2_f)
        np.save(root_folder + "DATApython/cam_2_test_X_"+str(person_name)+"_"+str(session_number)+".npy", cam_2_y)

        np.save(root_folder + "DATApython/cam_3_train_X_"+str(person_name)+"_"+str(session_number)+".npy", cam_3_f)
        np.save(root_folder + "DATApython/cam_3_test_X_"+str(person_name)+"_"+str(session_number)+".npy", cam_3_y)

        np.save(root_folder + "DATApython/cam_4_train_X_"+str(person_name)+"_"+str(session_number)+".npy", cam_4_f)
        np.save(root_folder + "DATApython/cam_4_test_X_"+str(person_name)+"_"+str(session_number)+".npy", cam_4_y)
	    
        np.save(root_folder + "DATApython/all_cams_train_X_"+str(person_name)+"_"+str(session_number)+".npy", all_cams_f)
        np.save(root_folder + "DATApython/all_cams_test_X_"+str(person_name)+"_"+str(session_number)+".npy", all_cams_y)
	    
        np.save(root_folder + "DATApython/labels_train_Y_"+str(person_name)+"_"+str(session_number)+".npy", labels_train_Y)
        np.save(root_folder + "DATApython/labels_test_Y_"+str(person_name)+"_"+str(session_number)+".npy", labels_test_Y)

        np.save(root_folder + "DATApython/movements_id_train_Y_"+str(person_name)+"_"+str(session_number)+".npy", movements_id_train_Y)
        np.save(root_folder + "DATApython/movements_id_test_Y_"+str(person_name)+"_"+str(session_number)+".npy", movements_id_test_Y)

    #now make database for tensor flow
