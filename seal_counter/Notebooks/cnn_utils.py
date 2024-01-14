import numpy as np
import torch

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Tuple

from sklearn.metrics import(
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    f1_score, 
    precision_score, 
    recall_score, 
)

# Class for Dataset object
class SealDataset(Dataset):
    def __init__(self, images, labels, transform=transforms.Compose([transforms.ToTensor()])):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        self.length = len(self.labels)
        return self.length

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if label == 0:
            ret_label = torch.FloatTensor([1, 0])
        else:
            ret_label = torch.FloatTensor([0, 1])
        return self.transform(image), ret_label
    

def get_labels_and_sub_images(img_data:np.array, bb_data:np.array, threshold:float=.3) -> Tuple[List, List, List, List]:
    """Takes total images and bounding box data and returns separate arrrays for sub-images with seals, labels for sub-images with seals, sub-image without seals, labels without seals 

    Args:
        img_data (np.array): An array containing image information. Images are of the shape (150, 150, 3)
        bb_data (np.array): An array containing bounding box information. If a sub-image has a seal, the corresponding index will contain a pandas DataFrame otherwise None
        threshold (float, optional): Seal threshold. Sub-images containing less seal will be filtered out. Defaults to .3.

    Returns:
        Tuple[List, List, List, List]: Returns sub-images with seals, labels of seals, sub-images without seals, and labels without seals respectively
    """

    # Only two classes, seal (1) or no seal (0)
    label_1_img = []
    label_1 = []
    label_0_img = []
    label_0 = []
    
    for idx in range(len(bb_data)):
        seal_data_frame = bb_data[idx]
        sub_image = img_data[idx]

        # Is a seal present in the data
        if seal_data_frame is not None:
            
            # Determine the maximum percent of seal present in a sub-image
            percent = max(seal_data_frame["percent_seal"])
            
            # If the max percent is above the seal threshold
            if percent > threshold:
                label_1_img.append(sub_image)
                label_1.append(1)

            # Even though there are seals, they are not counted
            else:
                label_0_img.append(sub_image)
                label_0.append(0)
        
        # No Seal
        else:
                label_0_img.append(sub_image)
                label_0.append(0)

    return label_1_img, label_1, label_0_img, label_0


def get_labels(bb_data:np.array, threshold:float=.3)-> List[int]:
    """Extracts labels for sub-images. Can filter based off the given threshold

    Args:
        bb_data (np.array): Array containing bounding box information
        threshold (float, optional): Seal Threshold value. Defaults to .3.

    Returns:
        List[int]: List of labels
    """
    labels = []
    for data_frame in bb_data:

        # Seal has been found
        if data_frame is not None:

            # See if the seal detected meets the threshold
            percent = float(
                data_frame
                .percent_seal
                .max()
                )
            if percent > threshold:
                labels.append(1)
            else:
                labels.append(0)
        
        # No Seal
        else:
            labels.append(0)
    
    return labels


def generate_predictions_pytorch(model, dataloader:DataLoader, device:str) -> Tuple[np.array, np.array]:
    """Generates actual labels and predicted labels for a given dataloader and model

    Args:
        model (torchvision model): Model which will generate predictions
        dataloader (DataLoader): DataLoader containing information
        device (str): Either 'cpu' or 'cuda' representing cpu or gpu usage respectively

    Returns:
        Tuple[np.array, np.array]: Numpy arrays for actual and predicted labels respectively
    """
    # Set model to evaluation
    model.eval()
    
    # Lists for actual and predicted labels 
    predicted_labels = []
    actual_labels = []

    for data, label in tqdm(dataloader):
        # Cast to device
        data = data.to(device)

        # Make prediction and remove from device
        raw_pred = (
            model(data)
            .cpu()
            .detach()
        )
        true_label = label

        # Get prediction from probability
        pred = np.argmax(raw_pred, axis=1)
        true_label = np.argmax(true_label, axis=1)

        # Store predited and actual label
        predicted_labels.append(pred.numpy())
        actual_labels.append(true_label.numpy())

    # Convert to single list
    actual_labels = np.concatenate(actual_labels, axis=0)
    predicted_labels = np.concatenate(predicted_labels, axis=0)

    return actual_labels, predicted_labels


def generate_predictions_tensor_flow(model, image_list:np.array) -> np.array:
    """Generate predictions using a tensorflow keras model

    Args:
        model (TensorFlow keras model): Model used to generate predictions
        image_list (np.array): List of sub-images to get predictions for

    Returns:
        np.array: Numpy array containing predicted labels
    """
    # Get raw probabilities for each sub-image
    raw_preds = (
        model
        .predict(image_list)
        .flatten()
        )
    
    # Convert probability into label
    predictions = np.array(
        [1 if pred > .5 else 0 for pred in raw_preds]
    )

    return predictions


def display_result_metrics(actual_labels:List[int], predicted_labels:List[int]) -> None:
    """Displays confusion matrix, accuracy, precision, recall, and f1-score given actual and predicted labels

    Args:
        actual_labels (List[int]): List containing the true label values
        predicted_labels (List[int]): List containing predicted label values
    """
    # Create confusion matrix
    cm_train = confusion_matrix(actual_labels, predicted_labels)
    ConfusionMatrixDisplay(confusion_matrix=cm_train).plot()

    # Calculate metrics
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels)
    recall = recall_score(actual_labels, predicted_labels)
    f1 = f1_score(actual_labels, predicted_labels)
    
    # Display metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")