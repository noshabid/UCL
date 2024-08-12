#######################################
# Unsupervised Curriculum Learning
# Implementation by Nosheen Abid
# Version 0.1, 20220425
#######################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
from datetime import datetime
from sklearn.cluster import KMeans
# import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import umap
import csv
from k_means_constrained import KMeansConstrained
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import SpectralClustering

from tensorflow.keras import regularizers #Need to be here if using sparse autoencoders
from tensorflow.keras.models import Model

import utilities  # this file must be in the same folder as the notebook
import autoencoders # this file must be in the same folder as the notebook
# 

def create_model(core_model, weights, input_shape, core_output_layer=None,
                 n_clusters=2, learning_rate=0.0001, momentum=0.9, random_seed=None):
    """
    Creates a model based on a pretrained model.

    :param core_model: The core model to base the model on.
    :param weights: Pretrained weights. Should usually be "imagenet" or None.
    :param input_shape: shape of input, for example (64, 64).
    :param core_output_layer: The final layer before the flattening layer. Use if you want to try outputting from
    an earlier layer.
    :param n_clusters: Number of categories to classify between.
    :param learning_rate: Learning rate of model.
    :param momentum: Momentum of model.
    :param random_seed: Random seed, if you like.
    :return: Model object.
    """

    core = core_model(weights=weights,
                      include_top=False,
                      input_tensor=tf.keras.layers.Input(shape=input_shape))
    if core_output_layer is not None:
        print(f"Setting {core_output_layer} as output layer for the core model (to feed into flatten layer.")
        x = core.get_layer(core_output_layer).output
    else:
        x = core.output
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dropout(0.5, name="final_dropout", seed=random_seed)(x)
    x = tf.keras.layers.Dense(n_clusters,
                              activation="softmax",
                              name="out_layer",
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_seed)
                              )(x)
    model = tf.keras.models.Model(inputs=core.input, outputs=x)

    # Note: Model will be recompiled anyway before learning, this is just to get rid of a warning message.
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model


def train_model(model, data, train_datagen, freeze_layers, learning_rate, momentum, n_epochs,
                batch_size=16, img_size=(112, 112), use_validation=False, random_seed=None):
    """
    Trains a model, updating its weights.

    :param model: Model to train.
    :param data: Data to train model with. (DataFrame)
    :param train_datagen: ImageDataGenerator to use.
    :param freeze_layers: Number of layers to freeze.
    :param learning_rate: Learning rate to use.
    :param momentum: Momentum to use.
    :param n_epochs: Number of epochs to train model.
    :param batch_size: Batch size to use.
    :param img_size: Image size to use, for example (64,64).
    :param use_validation: Whether to use a validation set or not.
    :param random_seed: Random seed, if you want reproducibility.
    :return: Trained model and history.
    """

    # Make train/validate split
    if use_validation:
        train, validate = train_test_split(data, train_size=0.8, shuffle=True, stratify=data['plabel'], random_state=random_seed)
    else:
        train = data
        validate = None

    print("**************************************************")
    print(train["plabel"].unique())
    print(validate["plabel"].unique())
    print("**************************************************")
    # Import data from directories and turn it into batches
    data_train = train_datagen.flow_from_dataframe(train,
                                                   x_col="path",
                                                   y_col="plabel",
                                                   batch_size=batch_size,
                                                   target_size=img_size,
                                                   class_mode="categorical",
                                                   shuffle=True,
                                                   seed=random_seed)
    if use_validation:
        data_validate = train_datagen.flow_from_dataframe(validate,
                                                          x_col="path",
                                                          y_col="plabel",
                                                          batch_size=batch_size,
                                                          target_size=img_size,
                                                          class_mode="categorical",
                                                          shuffle=True,
                                                          seed=random_seed)
    else:
        data_validate = None

    # Freeze some layers
    print("Freezing the first " + str(freeze_layers) + " layers (out of " + str(len(model.layers)) + ").")


    stop_train = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                    patience=2, min_lr=0.00001, verbose=1)
    
    if freeze_layers < 0:
        for layer in model.layers[:freeze_layers]:
            layer.trainable = False
        for layer in model.layers[freeze_layers:]:
            layer.trainable = True
        cb = []
    else:
        for nr, layer in enumerate(model.layers):
            if nr < freeze_layers:
                layer.trainable = False
            else:
                layer.trainable = True
            cb = [reduce_lr, stop_train]


    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])
    print("Training model...")
    hist = model.fit(data_train,
                     validation_data=data_validate,
                     verbose=0,
                     epochs=n_epochs,
                     callbacks=cb
                     )
    print(f"Finished training after {len(hist.history['loss'])} epochs.")

    return model, hist


def extract_features(data, model, img_datagen, batch_size=16, img_size=(112, 112), random_seed=None):
    """
    Runs data through a model, and returns a vector of the output features.

    :param data: Dataframe, or something that can be converted to a dataframe, containing paths to
        images in a column called "path".
    :param model: A model that must have a layer named "flatten", which will be used as output layer.
    :param img_datagen: ImageDataGenerator to use.
    :param batch_size: Batch size to use.
    :param img_size: Image size to use.
    :return: Array of features for every entry in data.
    """
    # Make sure it actually has a "path" column...
    if type(data) is not pd.core.frame.DataFrame:
        data = pd.DataFrame(data, columns=["path"])

    # Import data from directories and turn it into batches
    image_data = img_datagen.flow_from_dataframe(data,
                                                 x_col="path",
                                                 batch_size=batch_size,
                                                 target_size=img_size,
                                                 class_mode=None,  # no labels!
                                                 validate_filenames=True,
                                                 shuffle=False,
                                                 seed=random_seed)  # don't shuffle

    model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('flatten').output)

    # Run data through model
    print("Extracting features from data...", end="")
    features = model.predict(image_data)
    print("done! Features extracted. Shape:", features.shape)

    return features


def make_autoencoder(features, checkpoint_path, iteration, latent_dim=100, epochs=500, finetune=False):    
    """
    Takes feature matrix and returns a representation of them in a lower dimension.

    :param features: Matrix of extracted features.
    :param latent_dim: Number of dimensions that the features will be reduced to.
    :param iteration: Used for finetuning logic if enabled
    :param latent_dim: Size of fully connected latent dim layer in the autoencoder
    :param epochs: Max number of epochs to train the autoencoder
    :param finetune: Loads autoencoder saved from last iteration if True, otherwise trained from scratch
    :return: Array of encoded features for every entry in data.
    """
    
    #Shape of extracted features, used to set the output layer size of the autoencoder
    shape = features.shape[1:]

    #Create and compile the autoencoder
    #Check if finetune autoencoder or train from scratch
    if iteration <= 2 or finetune == False:
        autoencoder = autoencoders.Autoencoder(latent_dim, shape)
    else:
        autoencoder = tf.keras.models.load_model(checkpoint_path + "autoencoder.ckpt")

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=['mae'])

    #Callback - early stopping
    stop_train = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.00001)

    cb = [stop_train]

    print(f"Training autoencoder with latent_dim={latent_dim}...")
    hist = autoencoder.fit(features, features,
                    epochs=epochs,
                    batch_size=512,
                    shuffle=True,
                    validation_split=0.2,
                    callbacks=cb,
                    verbose=False
                    )
    print("Done!")
    autoencoder.save(checkpoint_path + "autoencoder.ckpt")
    
    #Run the features through the encoder and return the latent space encoding
    encoded_features = autoencoder.encoder(features).numpy()
    return encoded_features, hist

def make_umap(features,
              dim=100,
              random_state=None):
    """
    Takes feature matrix and returns a representation of them in a lower dimension.

    :param features: Matrix of extracted features.
    :param dim: Number of dimensions the features will be reduced to.
    :param random_seed: Random seed, for reproducibility.
    :return: Array of encoded features for every entry in data.
    """
    print(f"Reducing features to {dim} dimensions with UMAP...")
    features = umap.UMAP(
        n_neighbors=50,
        min_dist=0.0,
        n_components=dim,
        random_state=random_state,
        ).fit_transform(features)
    print("Done!")
    return features

def make_clusters(features,
                  n_clusters=2,
                  centroids=None,
                  random_seed=None):
    """
    Takes feature matrix and returns a K-means cluster object.

    :param features: Matrix of extracted features.
    :param n_clusters: Number of clusters.
    :param centroids: Optional starting centroids for each cluster.
    :param random_seed: Random seed, for reproducibility.
    :return: Cluster object.
    """
    # Cluster all data, based on the features
    print(f"Clustering features into {n_clusters} clusters with K-means...", end="")
    if centroids is None:  # random starting points for the first iteration
        print("using random clustering...", end="")
        clust1 = KMeansConstrained(n_clusters=n_clusters,
                       size_min=20,
                       verbose=0,
                       n_init=1, random_state=random_seed).fit(features)     
    else:  # use the centroids from the previous iteration
        print("using cluster centers from last iteration...", end="")
        clust1 = KMeansConstrained(n_clusters=n_clusters,
                       size_min=20,
                       verbose=0,
                       init=centroids,
                       n_init=1, random_state=random_seed).fit(features)        
    print("done!")
    
    
    return clust1

def make_clusters_spectral(features,
                  n_clusters=2,
                  random_seed=None
                  ):
    """
    Takes feature matrix and returns a spectral cluster object.

    :param features: Matrix of extracted features.
    :param n_clusters: Number of clusters.
    :param random_seed: Random seed, for reproducibility.
    :param encode: Use autoencoder for dimensionalityreduction
    :return: Cluster object.
    """

    # Cluster all data, based on the features
    print(f"Clustering features into {n_clusters} clusters with spectral clustering...", end="")
    clust1 = SpectralClustering(n_clusters=n_clusters,
                                affinity='nearest_neighbors',
                                assign_labels='kmeans',
                                verbose=False,
                                n_init=10,
                                n_jobs=-1,
                                n_neighbors=100,
                                random_state=random_seed).fit(features)     
    print("done!")
    
    return clust1


def make_ucl(model,
             data,
             true_labels=None,
             n_clusters=2,
             start_iter=1,
             stop_iter=10,
             starting_centers=None,
             freeze_centers=False,
             use_previous_centers=False,
             random_seed=None,
             elambda=0.8,
             batch_size=16,
             img_size=(112, 112),
             ext_datagen=None,
             train_datagen=None,
             freeze_layers=0,
             learning_rate=0.0001,
             momentum=0.9,
             n_epochs=10,
             use_validation=True,
             checkpoint_path="",
             write_logs=True,
             log_path="",
             plots_path="",
             write_plots=True,  # both images and figures
             show_images=True,
             show_figs=True,
             n_best_imgages_to_show=1,
             clustering_method="spectral",
             use_dim_red=True,
             umap_iterations=1,
             comment=""):
    """

    :param model: Model to use (would normally be created by the create model function above).
    :param data: Dataset to use. Should be a pandas DataFrame or series with paths.
    :param true_labels: True labels, if avaiable.
    :param n_clusters: Number of clusters/classes.
    :param start_iter: Starting iteration. Normally 1.
    :param stop_iter: Final iteration. Should be higher than start_iter.
    :param starting_centers: Starting cluster centers, to use if you want to assign fixed centers to the first iter.
    :param freeze_centers: Freeze the first n layers of the model before training
    :param use_previous_centers: Whether to use centers from previous iteration as centers, rather than randomizing.
    :param random_seed: Use for reproducibility.
    :param elambda: eLambda value to use. Should be a float between 0.0 and 1.0 or "auto".
    :param batch_size:
    :param img_size:
    :param ext_datagen:
    :param train_datagen:
    :param freeze_layers:
    :param learning_rate:
    :param momentum:
    :param n_epochs:
    :param use_validation:
    :param checkpoint_path:
    :param write_logs:
    :param log_path:
    :param plots_path:
    :param write_plots: Whether to save plots and images to file (using the plots_path).
                        Needs show_images and show_figs to be true.
    :param show_images:
    :param show_figs:
    :param n_best_imgages_to_show: Number of best images in each cluster to show.
    :param comment: Comment to include in log file.
    :return:
    """

    dt = datetime.now()
    timestamp = str(dt)[:str(dt).find(".")].replace("-", "").replace(":", "").replace(" ", "_")  # use for log file name

    # Parameters to put in log file
    params = f"""
start time= {str(dt.date()), str(dt.time())}
Comment: {comment}
----- -----    Parameters    ------ -----
data(shape)={data.shape}
n_clusters={n_clusters}
start_iter={start_iter}
stop_iter={stop_iter}
starting_centers={starting_centers}
freeze_centers={freeze_centers}
use_previous_centers={use_previous_centers}
random_seed={random_seed}
freeze_layers={freeze_layers}
elambda={elambda}
learning_rate={learning_rate}
momentum={momentum}
n_epochs={n_epochs}
use_validation={use_validation}
"""

    histories = []  # history objects from training
    pseudo_labels = []  # list of lists
    similarities = []  # all similarity scores for all iterations and all clusters
    cols = ["ss_dists", "purity", "silhouette", "reliable", "chgd_lbls"]
    for n in range(n_clusters):
        cols.append("n_in_" + str(n))
        cols.append("rel_in_" + str(n))
    cols.append("epochs")
    iter_metrics = pd.DataFrame(index=range(start_iter, stop_iter + 1), columns=cols)

    if write_logs:
        file = open(log_path + timestamp + ".log", "w")
        file.write(params + "\n")
        file.write("iter".ljust(5))
        for col in cols:
            file.write(col.rjust(11))
        file.write("\n")
        file.close()

    if starting_centers is not None:
        print("Using supplied starting cluster centers.")
        centroids = starting_centers
    else:
        centroids = None

    iteration = start_iter
    while iteration <= stop_iter:

        print(f"\n---- ITERATION {iteration} ----")

        rel = []  # stores number of reliable samples per cluster for logging

        # Run data through model indicated by checkpoint, returning features as an array
        features = extract_features(data, model=model, img_datagen=ext_datagen,
                                    batch_size=batch_size, img_size=img_size,
                                    random_seed=random_seed)

        if use_dim_red:
            if iteration <= umap_iterations:
                features_clustering = make_umap(features)
            else:
                features_clustering, hist_ae = make_autoencoder(features,
                                                                checkpoint_path=checkpoint_path,
                                                                iteration=iteration,
                                                                finetune=False)
                if show_figs:
                    plt.figure(figsize=(4, 2))
                    plt.plot(hist_ae.history["loss"])
                    if use_validation:
                        plt.plot(hist_ae.history["val_loss"])
                    plt.title("Autoencoder Loss")
                    if write_plots:
                        plt.savefig(plots_path + f"autoencoder_it{iteration}_training_loss.jpg", bbox_inches='tight')
                    plt.show()
        else:
            features_clustering = features


        if clustering_method == "kmeans":
            clust = make_clusters(features_clustering, n_clusters, centroids, random_seed)

            if not freeze_centers:
                print("Updating cluster centers.")
                # to use when initializing in the next iteration (in case use_previous_centroids=True)
                centroids = clust.cluster_centers_
            else:
                print("Keeping cluster centers from previous iteration.")

        elif clustering_method == "spectral":
            clust = make_clusters_spectral(features_clustering, n_clusters, random_seed)

            #Only used for cluster metrics, these will not be accurate when using spectral clustering
            clust_kmeans = make_clusters(features, n_clusters, centroids, random_seed)

            if not freeze_centers:
                print("Updating cluster centers.")
                # to use when initializing in the next iteration (in case use_previous_centroids=True)
                centroids = clust_kmeans.cluster_centers_
            else:
                print("Keeping cluster centers from previous iteration.")
        
        if elambda == "lof":
            # Step 3: Apply LOF on Each Cluster
            lof_labels = np.zeros_like(clust.labels_, dtype=int)
            for cluster_id in np.unique(clust.labels_):
                cluster_mask = (clust.labels_ == cluster_id)
                cluster_data = features[cluster_mask]

                if iteration < 2:
                    cont = 0.5
                else:
                    cont = 1/iteration

                # Apply LOF on each cluster
                lof = LocalOutlierFactor(n_neighbors=2, algorithm='auto', contamination=cont)  # Adjust parameters as needed
                cluster_lof_labels = lof.fit_predict(cluster_data)
                cluster_lof_labels[cluster_lof_labels == 1] = cluster_id

                # Combine LOF labels with k-means labels
                lof_labels[cluster_mask] = cluster_lof_labels
            pseudo_labels.append(lof_labels)  # predicted "labels", ie clusters
        else:
            pseudo_labels.append(clust.labels_)
        
        if clustering_method == "kmeans":
            distances = clust.transform(features)  # distances to each cluster center for every sample
            ss_distances = clust.inertia_  # sum of squares of distances in the cluster
        else:
            distances = clust_kmeans.transform(features)  # distances to each cluster center for every sample
            ss_distances = clust_kmeans.inertia_  # sum of squares of distances in the cluster

        # Compare labels between iterations, to see how many have changed
        if len(pseudo_labels) > 1:
            same_label = list(pseudo_labels[-2] - pseudo_labels[-1])  # 0 if they are the same in both iterations.
            changed_labels = len(same_label) - same_label.count(0)
            # Since cluster number is randomly allocated, we may need to switch the numbers if they are not
            # the same as the previous iteration. This code switches cluster numbers if more than half the
            # labels have changed. Note that there is no guarantee that the cluster number was "correct" (compared
            # to the true labels) in the first place, this just ensures that they are "wrong" in the same way every
            # iteration.
            # Only do this when there are only two clusters/labels. If there are more, it gets more complicated.
            if changed_labels > len(same_label)*0.5 and len(set(pseudo_labels[-1])) == 2:
                print("Inverting Pseudo Labels.")
                pseudo_labels[-1] = np.array(list(map(lambda x: abs(x-1), pseudo_labels[-1])))
                same_label = list(pseudo_labels[-2] - pseudo_labels[-1])
                changed_labels = len(same_label) - same_label.count(0)
        else:
            changed_labels = 0  # For the first iteration.

        if true_labels is not None:  # Only do this when there are actually any true labels to compare with
            purity = utilities.purity_score(tf.keras.utils.to_categorical(true_labels),
                                            tf.keras.utils.to_categorical(pseudo_labels[-1]))
        else:
            purity = 0

        silhouette_scores = silhouette_score(features, pseudo_labels[-1])

        center_idx = np.argmin(distances, axis=0)  # index of the "best" (most central) sample in each cluster
        centers = [features[ci] for ci in center_idx]  # features of the above samples

        # Calculate similarity matrix
        similarities.append(utilities.make_similarity_matrix(centers, features))

        reliable_imgs = [pd.DataFrame] * n_clusters
        predictions_df = utilities.make_predictions_df(data, true_labels, pseudo_labels, similarities)
        preds = [pd.DataFrame] * n_clusters

        # Selects the "best" samples from each cluster, putting
        # them into a dataframe.
        v = []
        
        if elambda == "lof":
            for r in range(n_clusters): 
                # only samples with that perticular cluster are considered and outliers are ignored automatically as they have label -1 which will 
                # never come in the loop
                preds[r] = predictions_df[(predictions_df["plabel"] == r)].copy()
                preds_count = len(predictions_df[(predictions_df["plabel"] == r)])
                v.append(int(preds_count))
            print(f'Reliable samples for each cluster is: \n{v}')
            n_reliable = min(v) 
            for r in range(n_clusters):
                preds[r] = preds[r].sample(frac=1).reset_index(drop=True)
                reliable_imgs[r] = preds[r][:n_reliable].copy()
                print(f"\tReliable samples: ", len(reliable_imgs[r]))
                rel.append(len(reliable_imgs[r]))
                # Show histogram of similarities.
                if show_figs:
                    plt.figure(figsize=(4, 2))
                    plt.hist(preds[r][f"sim_{r}"], bins=25, range=(0.0, 1.0))
                    plt.xlim([0, 1])
                    if write_plots:
                        plt.savefig(plots_path+f"it{iteration}_cl{r}_similarities.jpg", bbox_inches='tight')
                    plt.show()
        elif elambda == "auto":
            for r in range(n_clusters):
                preds_count = len(predictions_df[(predictions_df["plabel"] == r)])
                v.append(int(preds_count * 0.75)) # ensures at most 75% of cluster is "reliable"

            n_reliable = min(min(v),int(iteration * len(predictions_df) * 0.03)) #select a fixed value of lambda for all clusters
            
            
            for r in range(n_clusters):

                preds[r] = predictions_df[(predictions_df["plabel"] == r)].copy()
                preds[r] = preds[r].sort_values(by=f"sim_{r}", ascending=False)  # Cluster r sorted by similarity

                print(f"Cluster {r}:")
                print(f"\tNumber of samples:", len(preds[r]))
                print(f"\tSimilarity mean:  ", round(statistics.mean(preds[r][f"sim_{r}"]), 3),
                    "min/max:", round(min(preds[r][f"sim_{r}"]), 3), round(max(preds[r][f"sim_{r}"]), 3))

                if elambda == "auto":
                    # elambda "auto" sets number of reliable samples in each cluster to a fixed value, increasing
                    # every iteration. This makes both clusters have an equal number of reliable samples (limited by
                    # actual samples in the cluster)
                    # Number of reliable sample from cluster is the LOWEST of
                    #  *** [75% of all samples in cluster] and
                    #  *** [(iteration number)*(total number of samples in dataset) * 0.03].
                    reliable_imgs[r] = preds[r][:n_reliable].copy()
                else: 
                    # Put at least 5 samples as reliable, from each cluster. If cluster contains less than
                    # 5 samples, count entire cluster as reliable.
                    min_reliable = min(len(preds[r]), 5)
                    min_sim_to_include = preds[r][f"sim_{r}"].iloc[min_reliable-1]
                    reliable_imgs[r] = preds[r][(preds[r][f"sim_{r}"] >= min(elambda, min_sim_to_include))]

                print(f"\tReliable samples: ", len(reliable_imgs[r]))
                rel.append(len(reliable_imgs[r]))

                # Show histogram of similarities.
                if show_figs:
                    plt.figure(figsize=(4, 2))
                    plt.hist(preds[r][f"sim_{r}"], bins=25, range=(0.0, 1.0))
                    plt.xlim([0, 1])
                    if write_plots:
                        plt.savefig(plots_path+f"it{iteration}_cl{r}_similarities.jpg", bbox_inches='tight')
                    plt.show()
            # --- Per-cluster loop ends here. ---

        # Append all reliable samples to a single dataframe.
        all_reliable = pd.concat([reliable_imgs[0]])
        for cl in range(1, n_clusters):
            all_reliable = pd.concat([all_reliable, reliable_imgs[cl]])
        rel_count = len(all_reliable)
        
        #store reliable sample and clustering information
        df = pd.DataFrame()
        df['path'] = data
        df['true_labels'] = true_labels
        if elambda == 'lof':
            df['predictions'] = lof_labels
        else:
            df['predictions'] = clust.labels_
        df['distance0'] = distances[:,0]
        df['distance1'] = distances[:,1]
        df['sim_0'] = predictions_df['sim_0']
        df['sim_1'] = predictions_df['sim_1']
        df['reliable'] = False 
        df['reliable'][predictions_df['path'].isin(list(all_reliable['path']))] = True
        df.to_csv(plots_path+f"it{iteration}_clusters_info.csv")
      
        # Print  metrics for the current iteration.
        print("SS Distances:      " + str(ss_distances))
        print("Purity scores:     " + str(purity))
        print("Silhouette scores: " + str(silhouette_scores))
        print("Reliable images:   " + str(rel_count))
        print("Changed labels:    " + str(changed_labels))

        with open(log_path + 'purity.csv','a') as f1: 
            writer=csv.writer(f1)
            writer.writerow([purity])
            
        with open(log_path + 'silhouette_scores.csv','a') as f1: 
            writer=csv.writer(f1)
            writer.writerow([silhouette_scores])
            
        with open(log_path + 'ss_distances.csv','a') as f1: 
            writer=csv.writer(f1)
            writer.writerow([ss_distances])
            
        with open(log_path + 'changed_labels.csv','a') as f1: 
            writer=csv.writer(f1)
            writer.writerow([changed_labels])
            
        with open(log_path + 'rel_count.csv','a') as f1: 
            writer=csv.writer(f1)
            writer.writerow([rel_count])
        
        if clustering_method == "kmeans":
            umap_data = np.vstack([features, clust.cluster_centers_])
        else:
            umap_data = np.vstack([features, clust_kmeans.cluster_centers_])
        
        # create U-Map for visulaization of clusters
        standard_embedding = umap.UMAP(random_state=42).fit_transform(umap_data)
    
        plt.scatter(standard_embedding[:-n_clusters, 0], standard_embedding[:-n_clusters, 1], c=true_labels, s=0.3);
        plt.savefig(plots_path+f'it{iteration}_real_data_'+str(iteration)+'.png', dpi=300, bbox_inches='tight')
        plt.clf()
        
        plt.scatter(standard_embedding[:-n_clusters, 0], standard_embedding[:-n_clusters, 1], c=clust.labels_, s=0.3);
        plt.scatter(standard_embedding[-n_clusters:, 0], standard_embedding[-n_clusters:, 1], marker='^', s=0.5)
        plt.scatter(standard_embedding[center_idx, 0], standard_embedding[center_idx, 1], marker='s', s=0.5)
        plt.savefig(plots_path+f'it{iteration}_clustering_'+str(iteration)+'.png', dpi=300, bbox_inches='tight')
        plt.clf()
        
        # Show the most central images from each cluster, ordered by similarity. The first image is the centroid,
        # (since it, by definition, has a similarity score of 1.0 with itself).
        if show_images:
            for n in range(n_clusters):
                x = utilities.view_images(reliable_imgs[n]["path"],
                                          labels=reliable_imgs[n][f"sim_{n}"],
                                          n_images=n_best_imgages_to_show,
                                          # imgdatagen=ext_datagen,
                                          cmap="Greys",
                                          randomize=False,
                                          size=(3, 3))
                if write_plots:
                    x.savefig(plots_path+f"it{iteration}_cl{n}_most_similar.jpg", bbox_inches='tight')
                plt.show(x)
                # Code below shows the LEAST similar n images in the cluster, ordered by least similar first.
                # x = utilities.view_images(reliable_imgs[n]["path"].iloc[::-1],
                #                          labels=reliable_imgs[n][f"sim_{n}"].iloc[::-1],
                #                          n_images=n_best_imgages_to_show,
                #                          imgdatagen=ext_datagen,
                #                          cmap="Greys",
                #                          randomize=False,
                #                          size=(3, 3)))
                # if write_plots:
                #     x.savefig(plots_path+f"it{iteration}_cl{n}_least_similar.jpg", bbox_inches='tight')

        # previous_model = tf.keras.models.load_model(checkpoint) # checkpoint of previous iteration

        all_reliable = all_reliable[["path", "plabel"]]
        all_reliable["plabel"] = all_reliable["plabel"].astype(str)
        
        # Save updated model checkpoint before training, so the model matches the metrics we get.
        model.save(checkpoint_path + str(iteration) + ".ckpt")

        #tf.random.set_seed(random_seed)
        #train the last layer only with two epochs
        model, hist = train_model(model, all_reliable, freeze_layers=-1,
                                  train_datagen=train_datagen, learning_rate=learning_rate,
                                  momentum=momentum, n_epochs=2,
                                  batch_size=batch_size, img_size=img_size,
                                  use_validation=use_validation, random_seed=random_seed)
        
        #tf.random.set_seed(random_seed)
        model, hist = train_model(model, all_reliable, freeze_layers=freeze_layers,
                                  train_datagen=train_datagen, learning_rate=learning_rate,
                                  momentum=momentum, n_epochs=n_epochs,
                                  batch_size=batch_size, img_size=img_size,
                                  use_validation=use_validation, random_seed=random_seed)

        if show_figs:
            plt.figure(figsize=(4, 2))
            plt.plot(hist.history["accuracy"])
            if use_validation:
                plt.plot(hist.history["val_accuracy"])
            plt.title("Accuracy")
            if write_plots:
                plt.savefig(plots_path + f"it{iteration}_training_acc.jpg", bbox_inches='tight')
            plt.show()

            plt.figure(figsize=(4, 2))
            plt.plot(hist.history["loss"])
            if use_validation:
                plt.plot(hist.history["val_loss"])
            plt.title("Loss")
            if write_plots:
                plt.savefig(plots_path + f"it{iteration}_training_loss.jpg", bbox_inches='tight')
            plt.show()
        histories.append(hist)

        # Put metrics for current iteration in dataframe. Also, write it to log file.
        iter_values = [round(ss_distances, 0), purity, silhouette_scores, rel_count, changed_labels]
        for n in range(n_clusters):
            iter_values.append(list(pseudo_labels[-1]).count(n))
            iter_values.append(rel[n])
        iter_values.append(len(hist.history["accuracy"]))
        iter_metrics.loc[iteration] = iter_values

        if write_logs:
            file = open(log_path + timestamp + ".log", "a")
            file.write(str(iteration).ljust(5))
            for col in iter_values:
                file.write(str(round(col, 4)).rjust(11))
            file.write("\n")
            file.close()

        iteration += 1
    # --- Iteration loop ends here ---

    model.save(checkpoint_path + str(iteration) + ".ckpt")

    iter_metrics["reliable"] = iter_metrics["reliable"].astype(int)
    iter_metrics["chgd_lbls"] = iter_metrics["chgd_lbls"].astype(int)

    if write_logs:  # write finish time to log file
        file = open(log_path + timestamp + ".log", "a")
        dt = datetime.now()
        file.write(f"end time= {str(dt.date()), str(dt.time())}\n")
        file.close()
    print("Finished.")
    return model, iter_metrics, pseudo_labels, similarities, histories


def predict(model,
            data,  # for example X_test
            ext_datagen,
            labels=None,  # for example y_test
            batch_size=16,
            invert_predictions_if_needed=False,
            random_seed=None):
    """
    Predicts labels on a dataset using the supplied model. Predicts both using ordinary prediction
    and clustering prediction. Puts all predicted labels in a dataframe, together with the dataset (paths).

    :param model: A model.
    :param data: Data, in the form of paths to image files. pandas Series or dataframe format, or similar.
    :param labels: True labels, if any exist.
    :param ext_datagen: ImageDataGenerator object to process images.
    :param batch_size:
    :param invert_predictions_if_needed: Whether to invert predictions if less than half are correct.
    :param random_seed:
    :return: dataframe with predictions
    """
    # Extract image size and number of clusters from model
    img_size = model.layers[0].input_shape[0][1:3]
    n_clusters = model.layers[-1].output_shape[1]

    if labels is None:
        true_labels = [""] * len(data)
    else:
        true_labels = labels

    # Generate using the trained model to get probabilities
    probs = model.predict(ext_datagen.flow_from_dataframe(pd.DataFrame({"path": data, "label": [""] * len(data)}),
                                                          x_col="path",
                                                          y_col="label",  # labels are not used here
                                                          target_size=img_size,
                                                          batch_size=1,
                                                          shuffle=False))
    preds = np.argmax(probs, axis = -1)

    # Run data through model , returning features as an array
    features = extract_features(data, model=model, img_datagen=ext_datagen,
                                batch_size=batch_size, img_size=img_size,
                                random_seed=random_seed)

    # Cluster all data, based on the features
    clust = make_clusters(features, n_clusters, centroids=None, random_seed=random_seed)
    c_preds = clust.labels_

    # Creates a dataframe with predictions, probabilities and (if available) true labels for the test set.
    if labels is None:  # if no labels are provided
        preds_df = pd.DataFrame({"path": data, "prob_0": probs[:, 0], "prob_1": probs[:, 1],
                                 "p_pred": preds, "c_pred": c_preds})
    else:  # include true labels etc.
        preds_df = pd.DataFrame({"path": data, "true": true_labels, "prob_0": probs[:, 0], "prob_1": probs[:, 1],
                                 "p_pred": preds, "c_pred": c_preds})
        preds_df["correct_p"] = preds_df["true"] == preds_df["p_pred"]
        preds_df["correct_c"] = preds_df["true"] == preds_df["c_pred"]
        # if less than half the predictions are correct, assume that the clusters and true labels are mismatched,
        # and invert the predicted labels. (Only works when there are only 2 clusters)
        if invert_predictions_if_needed and n_clusters == 2 and preds_df["correct_p"].sum() < len(preds_df) / 2:
            print("Inverting cluster labels.")
            preds_df["p_pred"] = list(map(lambda x: abs(x - 1), preds_df["p_pred"]))
            preds_df["correct_p"] = preds_df["true"] == preds_df["p_pred"]
        if invert_predictions_if_needed and n_clusters == 2 and preds_df["correct_c"].sum() < len(preds_df) / 2:
            print("Inverting cluster labels.")
            preds_df["c_pred"] = list(map(lambda x: abs(x - 1), preds_df["c_pred"]))
            preds_df["correct_c"] = preds_df["true"] == preds_df["c_pred"]

    # difference between probs should indicate how certain the algorithm is of its prediction.
    # only applies to the model prediction, not the prediction using clustering (since it does not have
    # probability values).
    preds_df["distance"] = preds_df["prob_0"] - preds_df["prob_1"]

    return preds_df

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def find_labels(true_labels, preds):
    ary = confusion_matrix(true_labels, preds)
    df = pd.DataFrame(ary)
    gt_p = []
    assign_labels(df,gt_p)
    new_preds = np.array(preds)
    for label in gt_p:
            indexes = np.where(np.array(preds)==label[1])
            for idx in indexes:
                new_preds[idx] = label[0]
    return new_preds, gt_p

def assign_labels(df,gt_p):
    new_df = df
    gt = list(df.index)
    if(len(df.index) >= 1): # rows of df -> true label
        # print(gt) # rows of df -> true label
        # print(df)
        # print(df.columns) # columns of df -> predictions
        for i in gt:
            # print("true",i)
            # c = np.argmax(df.loc[i]) # locate max index in row i
            c = df.loc[i].idxmax()
            # r = np.argmax(df[c]) # locate max index in col c
            r = df[c].idxmax()
            # print(r,c)
            if i == r:
                gt_p.append(tuple([r, c]))
                # print(gt_p)
                new_df = new_df.drop([r]) # i and r are same. It is to delete row
                del new_df[c] # it is to delete column
                # print(new_df)
        assign_labels(new_df,gt_p)

def predict_model(model,
            data,  # for example X_test
            ext_datagen,
            labels=None,  # for example y_test
            batch_size=16,
            label_names = {},
            is_clustering = False,
            n_clusters = 2,
            img_size=(112, 112), 
            random_seed=None):
    """
    Predicts labels on a dataset using the supplied model. Predicts both using ordinary prediction
    and clustering prediction. Puts all predicted labels in a dataframe, together with the dataset (paths).

    :param model: A model.
    :param data: Data, in the form of paths to image files. pandas Series or dataframe format, or similar.
    :param labels: True labels, if any exist.
    :param ext_datagen: ImageDataGenerator object to process images.
    :param batch_size:
    :param is_clustering: Weather use k-means clustering to generate predictions or not.
    :param img_size: Image size to use, for example (64,64).
    :param random_seed:
    :return: dataframe with predictions
    """
    # # Extract image size and number of clusters from model
    # img_size = model.layers[0].input_shape[0][1:3]
    # n_clusters = model.layers[-1].output_shape[1]
    # print(n_clusters)
    if labels is None:
        true_labels = [""] * len(data)
    else:
        true_labels = labels

    if is_clustering==True:
        # Run data through model indicated by checkpoint, returning features as an array
        features = extract_features(data, model=model, img_datagen=ext_datagen,
                                            batch_size=batch_size, img_size=img_size,
                                            random_seed=random_seed)

        n_clusters = len(labels.unique())
        print("Number of classes in the test set: "+ str(n_clusters))
        # Cluster all data, based on the features
        clust = make_clusters(features, n_clusters, centroids, random_seed)

        preds = clust.labels_
    else:
        # Generate using the trained model to get probabilities
        probs = model.predict(ext_datagen.flow_from_dataframe(pd.DataFrame({"path": data, "label": [""] * len(data)}),
                                                            x_col="path",
                                                            y_col="label",  # labels are not used here
                                                            target_size=img_size,
                                                            batch_size=1,
                                                            shuffle=False))
        preds = np.argmax(probs, axis = -1)

    posible_preds, gt_p = find_labels(true_labels, preds)
    print(gt_p) # (gt, pred) in print
    
    # Creates a dataframe with predictions, probabilities and (if available) true labels for the test set.
    if labels is None:  # if no labels are provided
        preds_df = pd.DataFrame({"path": data,
                                 "p_pred": posible_preds})
    else:  # include true labels etc.
        preds_df = pd.DataFrame({"path": data, "true": true_labels,
                                "p_pred": posible_preds})
        preds_df["correct_p"] = preds_df["true"] == preds_df["p_pred"]
        # else:
        #     preds_df = pd.DataFrame({"path": data, "true": label_names[true_labels],
        #                             "p_pred": label_names[posible_preds]})
        #     preds_df["correct_p"] = preds_df["true"] == preds_df["p_pred"]
        #     print(preds_df)
    # difference between probs should indicate how certain the algorithm is of its prediction.
    # only applies to the model prediction, not the prediction using clustering (since it does not have
    # probability values).
    return preds_df
    
