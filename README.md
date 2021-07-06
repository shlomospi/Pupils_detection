# Pupils_detection

This project contains modules for preproccesing visual data, training, testing and converting (with TF-TRT) 
Tensorflow2 models for pupil center inferencing.

## Install
`git clone https://github.com/shlomospi/Pupils_detection.git`
- create venv  
- install requirments.txt  

## Inference Via TRT Engine

`python InferneceTRT.py -v <path/to/video>`

options:  
-a     :    int, if larger than 1, a running average of prediction of the last will be displayed  
--save : bool, if True, will save result in the saved model's dir  
-v     : path/to/video. Otherwise, will use camera.  
--load: 'choice' to choose saved model manually. Otherwise, will use model in "saved_model" folder.  
The script will prompt the user to choose a folder containing a saved model for inference.  
-o     : "preproccessed" to save frames after preproccessing.  
-t     : Threshold for preproccesing. For now, needed to be adjusted to the trained model version.  
-bin   : True for conversion of pixels to binary values after preproccessing, if thresholding is being used.  

## Inference

`python Inference.py -v <path/to/video>`

options:  
-a     :    int, if larger than 1, a running average of prediction of the last will be displayed  
--save : bool, if True, will save result in the saved model's dir  
-v     : path/to/video. Otherwise, will use camera.  
--load: 'choice' to choose saved model manually. Otherwise, will use model in "saved_model" folder.  
The script will prompt the user to choose a folder containing a saved model for inference.  
-o     : "preproccessed" to save frames after preproccessing.  
-t     : Threshold for preproccesing. For now, needed to be adjusted to the trained model version.  
-bin   : True for conversion of pixels to binary values after preproccessing, if thresholding is being used.  

## Training

For training, first add to project's directory folders of the dataset:  

images_dataset  
images_from_videos1_dataset  
images_from_videos3_dataset  
images_from_camera1_dataset  
IR videos  

alternativly, supply with new datasets and load them  

`python Main.py`

-p: Retraining will ask for a "saved_model" file and ignore any architecture choices  
-e: The maximal number of epochs to run  
-b: The size of a batch  
--lr: learning rate  
--rlr: by how much to reduce learning rate  
--log_dir: Directory name to save training logs, any pictures and saved models  
--blocks: num of blocks for a block type architecture  
--arch: small/blocks/medium  
-bin: -bin True' for converting data to binary pixels, ignore for False  
--arch: what dataset to load (IR/RGB/Both/BothRGB)  
--treshold: threshold (Hmin, Hmax, Smin, Smax, Vmin, Vmax) for image preproccessing  
--filters: filters for "medium" net  
-a: what augmentation to do? flip for flipingLR trans and an int for translating horizontaly up to "int" 
                             and vertically up to 2*"int"  
                             
The script will prompt the user to choose a folder containing a saved model if the phase is 'retrain'.  

## Preproccess Experimenting  

A simple GUI is available to eperiment with various preproccesses.  
The GUI runs a video in loops, while implementing Thresholding via bars.  
Based on the code provided at:   
https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html  

To run:  

`python Threshold_experimenting.py -v <relative/path/to/video>`

With options:

-v :name of video file to experiment on  
-r : resolution change of the input video. with be resized back for display  
-t : Threshold (Hmin, Hmax, Smin, Smax, Vmin, Vmax)  
-bin: '-bin True' for converting data to binary pixels, ignore for False   

## Convert Model to TRT Engine

Run:

`python TRTconverter.py`

with options:

-p : '16 for float16, 32 for float 32'

The script takes the model from the "saved_model" folder,  
converts it and saves it in the working directory