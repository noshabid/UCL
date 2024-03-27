import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import tensorflow as tf
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
import sklearn
import numpy as np
import matplotlib.image as mpimg
import random

def make_path_df(files, shuffle=False, random_state=42):
    """
    Takes a single filename, or a list of filenames, of text files containing rows with paths to image files + labels.
    Note: no control that the paths in the files are actually pointing to anything is performed.
    
    Args:
        files: A single string containing a filename, or a list of strings containing filenames.
        shuffle: Whether to shuffle the dataframe before returning it.
        random_state: Random state for shuffling.
        
    Returns a dataframe with paths and labels in separate columns.
    """
    # in case only a filename is sent, not as a list...
    if type(files) is str:
        files = [files]
        
    # first, put all lines from all files in a list object
    all_paths = []
    try:
        for i in range(len(files)):
            f = open(files[i], "r")
            all_paths = all_paths + f.readlines()
            f.close()
    except FileNotFoundError:
        print("File not found:x " + files[i])
        return

    # then convert the list to a dataframe with separate columns for paths and labels    
    df = pd.DataFrame(all_paths, columns=["path_label"])
    df[["path", "label"]] = df["path_label"].str.rsplit(pat=' ', n=1, expand=True)
    df = df.drop(columns=["path_label"])
    df["label"] = df["label"].str.strip() # strip newlines
    
    if df["path"].isnull().values.any():
        print("Warning: There are null values in the paths column. It's likely that something went wrong.")
    if df["label"].isnull().values.any():
        print("Warning: There are null values in the labels column. It's likely that something went wrong.")
    if shuffle:  # typically better to shuffle when making the train/test split instead...
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.

    Args:
        y_true: true labels as a 1D array
        y_pred: predicted labels as a 1D array

    Returns accuracy, precision, recall and f1-score in a dictionary.
    """
    # Calculate model accuracy
    accuracy = accuracy_score(y_true, y_pred)  # * 100
    # Calculate model precision, recall and f1 score using weighted average
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred,
                                                                     average="weighted", zero_division=0)
    results = {"accuracy": accuracy,
               "precision": precision,
               "recall": recall,
               "f1": f1}
    return results


def make_classification_report(y_true, y_preds, labels=("0", "1"), decimals=4, returndict=False):
    """
    Makes a report of classification metrics, with a customizable number of decimal points
    
    Args:
        y_true: true labels as a 1D array
        y_preds: predicted labels as a 1D array
        labels: list of class labels as strings
        decimals: number of decimal points to print out
        returndict: whether to return the report as a dictionary
    
    Returns report as a string object, unless returndict=True, in which case a dictionary is returned instead.
    """
    rep = sklearn.metrics.classification_report(y_true,
                                                np.round(y_preds),
                                                output_dict=True,
                                                zero_division=0,  # in case of zero div, return 0 value
                                                target_names=labels
                                                )
    if returndict:
        return rep  # returns standard dictionary
    
    dpt = decimals
    p_out  = "+------------- CLASSIFICATION REPORT -------------+\n"
    p_out += " accuracy " + str(round(rep["accuracy"], dpt)) + "\n"
    p_out += "               Precision Recall  F1-score Support\n"

    for lb in labels:
        p_out += (" " + str.ljust(lb, 14))
        for key in rep[lb]:
            if key == "support":
                p_out += (" " + str.ljust(str(round(rep[lb][key], dpt)), dpt+2, " ") + "  ")
            else:
                p_out += (" " + str.ljust(str(round(rep[lb][key], dpt)), dpt+2, "0") + "  ")
        p_out += "\n"

    p_out += (" " + str.ljust("macro avg", 14))
    for key in rep["macro avg"]:
        if key == "support":
            p_out += (" " + str(rep["macro avg"][key]) + "  ")
        else:
            p_out += (" " + str.ljust(str(round(rep["macro avg"][key], dpt)), dpt+2, "0") + "  ")
    p_out += "\n"
    p_out += (" " + str.ljust("weighted avg", 14))
    for key in rep["weighted avg"]:
        if key == "support":
            p_out += (" " + str(rep["weighted avg"][key]) + "  ")
        else:
            p_out += (" " + str.ljust(str(round(rep["weighted avg"][key], dpt)), dpt+2, "0") + "  ")
    return p_out


# from Nosheen
def purity_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = sklearn.metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# Calculate similarity matrix
def make_similarity_matrix(centers, features):
    """

    :param centers: matrix of features for the single most central sample in each cluster
    :param features: matrix of features for all samples
    :return: matrix of similarity scores for all samples
    """
    center_t_norm = np.float64(sklearn.preprocessing.normalize(centers))
    other_t_norm = np.float64(sklearn.preprocessing.normalize(features))

    print("\n")
    print(type(center_t_norm))
    print(type(other_t_norm))
    print("\n")
    print(center_t_norm[0])
    print(type(center_t_norm[0]))
    print(type(other_t_norm[0]))

    print("\n")
    print(center_t_norm[0,0])
    print(type(center_t_norm[0,0]))
    print(type(other_t_norm[0,0]))
    return tf.matmul(center_t_norm, other_t_norm, transpose_a=False, transpose_b=True)


def view_images(paths, labels=[""], n_images=1, imgdatagen=None, cmap=None, randomize=False, size=(4, 4)):
    """
    Shows a grid of n_images images.
    
    Args:
        paths: either a path as a string, or a list or dataframe containing paths
        labels: list or dataframe containing labels
        n_images: the number of images to show.
        imgdatagen: an optional imagedatagenerator to process the images through.
        cmap: optional color map.
        randomize: whether to show random images from the paths. Note: duplicates may occur.
        size: the size of it all.
        
    """

    if randomize:
        paths = paths.sample(n_images)
        
    if type(paths) == str: # in case it's only one,
        paths = [paths]  # make it a list of length 1
    
    # if there are no labels, fill with empty labels
    if (type(labels) is list) and (labels == [""]):
        labels = [""]*len(paths)
         
    if (type(paths) is pd.core.frame.DataFrame) and ("path" in paths.columns) and ("label" in paths.columns):
        labels = pd.DataFrame(paths["label"])
        paths = pd.DataFrame(paths["path"])
    else:
        labels = pd.DataFrame(labels)
        paths = pd.DataFrame(paths)

    if len(paths) < n_images:
        n_images = len(paths)

    # make a square grid for the images
    nrow = math.ceil(n_images**0.5)
    ncol = math.ceil(n_images**0.5)
    plt.figure(figsize=(ncol*size[0], nrow*size[1]))

    for i in range(n_images):
        plt.subplot(nrow, ncol, i+1)
        if randomize:
            n = random.randint(0, len(paths)-1)
        else:
            n = i

        img = mpimg.imread(paths.iloc[n][0])
        if imgdatagen is not None:
            imgarr = img_to_array(img)
            img = imgdatagen.standardize(np.copy(imgarr))
            # img = imgdatagen.random_transform(img)
        plt.imshow(img, cmap=cmap)
        plt.title(str(labels.iloc[n][0]), fontsize = 28)
        plt.axis('off')
    # plt.show()
    return plt


def make_predictions_df(paths, labels, plabels, similarities, iteration=-1):
    """
    Creates a dataframe containing paths, labels, predicted labels and similarities for each cluster.

    :param paths: Dataframe or Series containing paths to images.
    :param labels: True labels for images.
    :param plabels: Predicted pseudo-labels for images.
    :param similarities: Similarity values for each label (this is assumed to be a tensor).
    :param iteration: The iteration to pick plabels and similarities from. Defaults to the final iteration.
    :return: Dataframe
    """
    dfr = pd.DataFrame(paths)
    dfr["label"] = labels
    dfr["plabel"] = plabels[iteration]
    for n in range(len(similarities[iteration])):
        dfr[f"sim_{n}"] = similarities[iteration][n].numpy().tolist()
        dfr[f"sim_{n}"] = dfr[f"sim_{n}"].round(3)
    dfr["gap"] = abs(dfr["sim_0"] - dfr["sim_1"]).round(3)
    return dfr
