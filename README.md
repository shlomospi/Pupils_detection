## Pupils_detection

## install

- clone the repo and install requirments.txt

## Usage

# inference

python Inference.py -v <relative/path/to/video>

options:
-a :    int, if larger than 1, a running average of prediction of the last <a> will be displayed
--save: bool, if True, will save result in the saved model's dir

The script will prompt the user to choose a folder containing a saved model for inference.

# training

For training, folders of images and videos are needed to be added to the directory:
images_dataset
images_from_videos1_dataset
IR videos
alternativly, supply with new datasets and load them

python Main.py

-p: Retraining will ask for a "saved_model" file and ignore any architecture choices
-e: The maximal number of epochs to run
-b: The size of a batch
--lr: learning rate
--reducelr: by how much to reduce learning rate
--log_dir: Directory name to save training logs, any pictures and saved models
--blocks: num of blocks for a block type architecture
--arch: small/blocks/medium
-bin: -bin True' for converting data to binary pixels, ignore for False 
--arch: what dataset to load (IR/RGB/Both/BothRGB)
--treshold: threshold (Hmin, Hmax, Smin, Smax, Vmin, Vmax) for image preproccessing
--filters: filters for "medium" net
-a: what augmentation to do? flip for flipingLR trans and an int for translating horizontaly up to <int> 
                             and vertically up to 2*<int>
                             
The script will prompt the user to choose a folder containing a saved model if the phase is 'retrain'.
