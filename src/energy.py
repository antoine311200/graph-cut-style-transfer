import torch
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from maxflow.fastmin import aexpansion_grid
import src.graph_cut as gc

def cluster_style(style_features, k=3):
    """Cluster the style features using k-means.

    Select areas of the style features that are similar to each other and
    compute the cluster centers.
    Thus from a style features of shape (channel, height, width), we cluster based
    on the height and width and get the cluster centers of shape (k, channel).

    Args:
        style_features (np.array): Style features of the style image
        k (int, optional): Number of clusters. Defaults to 3.

    Returns:
        np.array: Cluster centers
        list: List of clusters
    """
    style_features_view = style_features.reshape(
        style_features.shape[0], -1
    ).T  # (height * width, channel)
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(style_features_view)

    cluster_centers = kmeans.cluster_centers_  # (k, channel)
    cluster_list = [style_features_view[kmeans.labels_ == i] for i in range(k)]
    return cluster_centers, cluster_list, kmeans.labels_


def data_energy(content_features_, cluster_centers,distance="cosine"):
    """Compute the data energy defined as 1 - cosine similarity between the content features and the cluster centers.

    Minimizing this energy term encourages the labeling operated with the KMeans to be consistent with the content features.

    Args:
        content_features (np.array): Content features of the content image
        cluster_centers (np.array): Cluster centers

    Returns:
        np.array: Data energy
    """
    content_shape = content_features_.shape # (channel, height, width)
    content_features = content_features_.reshape(content_shape[0], -1).T # (height * width, channel)

    if distance=="cosine":
        # Compute the cosine similarity between the content features and the cluster centers
        # Shapes: content_features (height * width, channel), cluster_centers (k, channel)
        # Result: similarity (height * width, k)
        similarity = cosine_similarity(content_features, cluster_centers)
        similarity = similarity.reshape(content_shape[1], content_shape[2], -1)
        distances = 1 - similarity

    else: #euclidean distance
        distances = np.linalg.norm(content_features[:, None, :] - cluster_centers[None, :, :], axis=2)
        distances = distances.reshape(content_shape[1], content_shape[2], -1)

    return distances


def smooth_energy(cluster_centers, lambd):
    """Compute the smooth energy defined as lambd * (1 - I) where I is the identity matrix.

    As the spatial content information is not taken into acontent_channelount in the clustering,
    failing to preserve discontinuity and producing unplesing structures, we add a smooth energy term
    to help pixels in the same content local region to have the same label.

    Args:
        cluster_centers (np.array): Cluster centers
        lambd (float): Weight of the smooth energy

    Returns:
        np.array: Smooth energy
    """
    k = cluster_centers.shape[0]
    return lambd * (1 - np.eye(k))


def total_energy(content_features, style_features, k=2, lambd=0.1, expansion="pymax", distance="cosine", max_cycles=30):
    """Compute the total energy of the graph cut algorithm.

    The total energy is the sum of the data energy and the smooth energy.

    Args:
        content_features (np.array): Content features of the content image
        style_features (np.array): Style features of the style image
        k (int, optional): Number of clusters. Defaults to 2.
        lambd (float, optional): Weight of the smooth energy. Defaults to 0.1.

    Returns:
        np.array: Total energy
    """
    cluster_centers, cluster_list, cluster_labels = cluster_style(style_features, k=k)

    data_term = data_energy(content_features, cluster_centers, distance=distance)  # (height, width, k)

    if expansion == "pymax":
        smooth_term = smooth_energy(cluster_centers, lambd)  # (k, k)

        data_term = data_term.astype(np.double)
        smooth_term = smooth_term.astype(np.double)

        labels = aexpansion_grid(data_term, smooth_term, max_cycles=None) # (height, width)
    else:
        greedy_assignments = np.argmin(data_term,axis=2) # (height, width)
        labels, energy, total_energy = gc.alpha_expansion(data_term, greedy_assignments, max_cycles=max_cycles, lambd=lambd)

    return labels, cluster_list, cluster_labels


def feature_WCT(content, style, epsilon=1e-8, alpha=1):
    """Compute the whitening and coloring transform for the content features based on the paper
    "Universal Style Transfer via Feature Transforms" by Li et al."

    Args:
        content_features (np.array): Content features of the content image
        style_features (np.array): Style features of the style image
        label (np.array): Label of the cluster
        alpha (float): Weight of the content features

    Returns:
        np.array: Transformed content features
    """
    content_channel, content_length = content.shape
    style_channel, style_length = style.shape

    content_mean = np.mean(content, axis=1, keepdims=True)
    content = content - content_mean
    content_covariance = np.matmul(content, content.T) / (content_length - 1) + np.eye(content_channel) * epsilon

    style_mean = np.mean(style, axis=1, keepdims=True)
    style = style - style_mean
    style_covariance = np.matmul(style, style.T) / (style_length - 1) + np.eye(style_channel) * epsilon

    content_U, content_S, _ = np.linalg.svd(content_covariance)
    style_U, style_S, _ = np.linalg.svd(style_covariance)

    content_sum = np.sum(content_S > 1e-5)
    style_sum = np.sum(style_S > 1e-5)
    content_U = content_U[:, :content_sum]
    style_U = style_U[:, :style_sum]

    content_D = np.diag(np.power(content_S[:content_sum], -0.5))
    style_D = np.diag(np.power(style_S[:style_sum], 0.5))

    whitening_matrix = content_U @ content_D @ content_U.T
    coloring_matrix = style_U @ style_D @ style_U.T

    result = coloring_matrix @ whitening_matrix @ content + style_mean

    result = alpha * result + (1 - alpha) * (content + content_mean)
    return result


def style_transfer(content_features, style_features, alpha=0.6, k=3, lambd=0.1, distance="cosine", expansion="pymax",max_cycles=30):
    """Perform the style transfer using the graph cut algorithm.

    Args:
        content_features (np.array): Content features of the content image
        style_features (np.array): Style features of the style image
        alpha (float, optional): Weight of the content features. Defaults to 0.6.
        k (int, optional): Number of clusters. Defaults to 3.
        lambd (float, optional): Weight of the smooth energy. Defaults to 0.1.

    Returns:
        np.array: Transfered features
    """
    # Shape of content_features and style_features: (channel, height, width)
    labels, cluster_list, cluster_labels = total_energy(content_features, style_features, k=k, lambd=lambd, expansion=expansion, distance=distance, max_cycles=max_cycles)
    labels = labels.flatten()

    transfered_features = np.zeros(content_features.shape).reshape(content_features.shape[0], -1)
    for i in range(k):
        # Check if the cluster does not have only a single pixel (no need to transfer)
        if cluster_list[i].shape[0] == 1:
            continue

        labels_idx = np.argwhere(labels == i).flatten()
        subcontent_features = content_features.reshape(content_features.shape[0], -1)[:, labels_idx]

        cluster_idx = np.argwhere(cluster_labels == i).flatten()
        subcontent_style = style_features.reshape(style_features.shape[0], -1)[:, cluster_idx]

        transfered_features[:, labels_idx] = feature_WCT(subcontent_features, subcontent_style, alpha=alpha)

    transfered_features = transfered_features.reshape(content_features.shape)
    transfered_features = torch.from_numpy(transfered_features).float()
    return transfered_features