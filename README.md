
WORK IN PROGRESS

torch installation: https://pytorch.org/get-started/locally/

# Seal Counting Thesis
Part of Kaanan Kharwa's Master Program at California Polytechnic State University, San Luis Obispo. The goal of this thesis is the lay the ground work for a tool to automatically count the number of seals from an aerial image of the beach (as pictured below).
![DJI_0037](https://github.com/kaanan99/Seal-Counting-Thesis/assets/55602809/50b74548-ed4a-4dae-93f9-42c030dab546)

## System Design
The following workflow is applied to aerial beach images:<br>
1. **Image Splitting:** Disassemble the original beach aerial image into smaller uniform sub-images <br>
2. **Seal Detection:** Recognize sub-images containing seals and discard those without any <br>
3. **Seal Counting:** Identify and count seals. The aggregated count across all sub-images produces the final seal count for the aerial beach image.

## How to use this repository
1. Clone this repository onto the f35 server <br>
2. Build and run docker container <br>
3. Open Jupyter Lab<br>

## Docker
When the docker container is run, the data will be copied over from the f35 server into the container.
[Put docker instructions here]

## Data location
Data is stored on the f35 server at the location `//data2/noses_data/seal_counting_thesis_data`

### CNN Training Files
The following files are used to train the CNN and water detector:
* `train_bb_data.npy`: Contains bounding box information for training sub-images.
* `train_images.npy`: Contains traning sub-images.
* `val_bb_data.npy`: Contains bounding box information for validation sub-images.
* `val_images.npy`: Contains validation sub-images.
* `test_bb_data.npy`: Contains bounding box information for testing sub-images.
* `test_images.npy`: Contains testing sub-images.
* `water_bb_data.pkl`: Contains bounding box information for where water is in sub-images.
* `water_images.pkl`: Contains sub-images for training the water classifier.

### Full images and XML files
The `Training, Val, and Testing Images` directory contains the full aerial images with their corresponding XML files for a specific dataset. The XML file contains all bounding box information for that image. There are 3 sub-directories, one for each dataset:
* Training (50 images)
* Validation (32 images)
* Testing (13 images)

## Contents
This section describes the contents of each directory.

### Seal Counter
This directory contains all code relevant to the seal counter component.
* **Data:** Contains sub-images for training seal and water detectors.
* **Generated Data:** Contains sub-images predicted to contain a seal by the specified model.
* **Models:** Trained models will be stored here.
* **Notebooks:** Contains all notebooks.

#### Seal Counter Notebooks
* `cnn_prediction_generation.ipynb`: Generates predictions for sub-images containing seals. Stores predictions in **Generated Data**.
* `cnn_utils.py`: Utility functions used throughout the directory.
* `evaluate_cnn_pytorch.ipynb`: Evaluates the performance of a Pytorch implemented CNN.
* `evaluate_cnn_tensorflow.ipynb`: Evaluates the performance of a TensorFlow implemented CNN.
* `train_model_pytorch.ipynb`: Trains a CNN to detect seals using Pytorch.
* `train_model_tensorflow.ipynb`: Trains a CNN to detect seals using Tensorflow.
* `water_classifier.ipynb`: Trains the water classifier.

### Image Splitting
Contains code to generate the RCNN training data
* `rcnn_training_data_generation.ipynb`: Parses over and splits the image based on the specified parameters and creates the RCNN training data. It is stored in `seal_counter/Data`

### Seal Detector
This directory contains all code relevant to the seal detector component.
* **Data:** Contains sub-images for training the RCNN.
* **Generated Data**: Contains predictions generated by the RCNN.
* **Models:** Trained models will be stored here.
* **Notebooks:** Contains notebooks for RCNN and Score predictor.

#### RCNN Notebooks
* `rcnn_utils.py`: A utility file containing helper functions for training the RCNN.
* `train_rcnn.ipynb`: Trains the RCNN. The trained model can be stored in the **Models** directory
* `evaluating_rcnn.ipynb`: Evaluates the performance of the specified RCNN.
* `generate_rcnn_predictions.ipynb`: Generates predictions for the training, validation, and testing datasets for the specified RCNN model.

#### Score Predictor Models
Many of the notebooks generate data needed to train the score predictor. It is recommended to run the notebooks in the order they are listed here.
* `generate_centriods.ipynb`: Generates centroid information for each dataset. Creates and stores information in a directory named **Centriods**.
* `grid_search.ipynb`: Conducts grid search and saves information. Creates and stores information in a directory named **Grid Search DataFrames**.
* `generate_score_predictor_dataframes.ipynb`: Combines centroid and grid search information into a single data frame used for training the score predictor. Creates and stores information in a directory named **Score Predictor DataFrames**.
* `score_predictor.ipynb`: Trains and evaluates score predictor.

### System Evaluation
Contains a notebook to evaluate the system by comparing actual and predicted counts.
* `evaluate_count.ipynb`: Evaluates the counts for the training, validation, and testing dataset. Loads RCNN models from `seal_counter/Models`.
