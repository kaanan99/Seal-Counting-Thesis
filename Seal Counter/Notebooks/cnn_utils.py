import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_labels_and_sub_images(img_data, bb_data, threshold=.3):
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


def get_labels(bb_data, threshold):
    labels = []
    for i in range(len(bb_data)):
        if bb_data[i] is not None:
            percent = float(max(bb_data[i]["percent_seal"]))
            if percent > threshold:
                labels.append(1)
            else:
                labels.append(0)
        else:
            labels.append(0)
    return np.array(labels)

def display_result_metrics(actual_labels, predicted_labels):
    cm_train = confusion_matrix(actual_labels, predicted_labels)
    ConfusionMatrixDisplay(confusion_matrix=cm_train).plot()
    tn = cm_train[0][0]
    tp = cm_train[1][1]
    fn = cm_train[1][0]
    fp = cm_train[0][1]

    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    print("Accuracy:", (tn + tp) / (tn + tp + fn + fp))
    print("Precision:", precision)
    print("Recall:", recall)