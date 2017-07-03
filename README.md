# Human Skeleton Tracking 

File and tools for the HPE17 dataset. This dataset consists of human  movements  acquired  using  the  Dynamic  and Active-pixel  Vision  Sensor  (DAVIS). The      database      includes synchronized   recording   from the a Vicon   Motion   Capture system, which  provides  a  ground  truth for the Human Skeleton 3d Position.  

Matlab Scripts: generates movies and labels
------

The matlab scripts are used to generates frames from DVS data. These frames are generated counting a fixed number of events. 
When running pyHumanSkeleton/matlab/DataBaseCreate.m the result are a series of movX.avi with movX.xml files.

The movX.avi contains frames with a fixed number of events, while the movX.xml contains the labels.
The movX.xml files contains labels for every frame in the movie. The labels are the average timestamp of the events, and a matrix of 13x3. This matrix represents the 3D position of  13 joints of the Human Skeleton. 
Every row is the 3D position of head, shoulder right, shoulder left, elobow right, elbow left, hip right, hip left, hand right, hand left, knee right, knee left, foot right, and foot left.

<mov type_num="1.xml" type_id="opencv-matrix">
 <timestamp>1166521</timestamp>
  <rows>13</rows>
  <cols>3</cols>
  <dt>d</dt>
    <data>
      74.530106 154.748535 1742.457397 
            -111.806908 233.553558 1474.824463 209.465820 
            260.723846 1463.713135 -178.777832 310.737030 
            1127.433716 269.367584 345.003723 1121.088379 
            -63.625969 193.162140 1031.098389 167.996643 
            196.067200 1018.562195 -207.448624 140.853470 
            887.212891 308.443146 199.028030 858.408813 
            -62.842102 36.649784 574.796326 118.969299 
            214.584305 538.524902 -131.301331 132.028351 
            76.007568 122.724083 144.655899 80.791191   
   </data>
</mov>
            
Python Scripts: prepare data for training
------

After the data have been pre-processed with the matlab scripts (mov and xml files have been generated). We can use the python tools to prepare the data before training a deep-network. This step includes organizing the data in batches and shuffling the movements. 

Have a look at the files:
python/data_formating_action_recognition.py
python/shuffle_data.py
python/data_preparation_human_recording.py

Python Scripts: train deep network with Tensorflow
------

A first example of Convolutional neural network has been scketched in the file:

python/cnn_training.py

This will train a simple CNN that will be able to carry action recognition among 33 classes of different actions.


