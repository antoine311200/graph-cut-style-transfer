import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from maxflow.fastmin import aexpansion_grid


def cluster_style(style_features, k=3):
    style_features_view = style_features.reshape(
        style_features.shape[0], -1
    ).T  # (channel, height * width)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(style_features_view)

    cluster_centers = kmeans.cluster_centers_  # (k, channel)
    cluster_list = [style_features_view[kmeans.labels_ == i] for i in range(k)]
    return cluster_centers, cluster_list


def data_energy(content_features, cluster_centers):
    content_shape = content_features.shape
    content_features = content_features.reshape(content_shape[0], -1).T

    similarity = cosine_similarity(content_features, cluster_centers)
    similarity = similarity.reshape(content_shape[1], content_shape[2], -1)

    return 1 - similarity


def smooth_energy(cluster_centers, gamma):
    k = cluster_centers.shape[0]
    return gamma * (1 - np.eye(k))


def total_energy(content_features, style_features, k=2, gamma=0.1):
    cluster_centers, cluster_list = cluster_style(style_features, k=k)

    data_term = data_energy(content_features, cluster_centers)  # (height, width, k)
    smooth_term = smooth_energy(cluster_centers, gamma)  # (k, k)

    data_term = data_term.astype(np.double)
    smooth_term = smooth_term.astype(np.double)

    labels = aexpansion_grid(data_term, smooth_term, max_cycles=None)

    return labels, cluster_list

def feature_WCT(content_features, style_features, label, alpha):
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
    channels = content_features.shape[0]
    cluster_size = style_features.shape[0]

    # Compute the mean of the content features
    # Multiply each channel by the label to put to zero the non-content features
    content_mask = content_features * label # (channel, height, width)
    content_mean = np.mean(content_mask, axis=(1, 2), keepdims=True) * label
    content_features = content_features - content_mean
    content_covariance = np.einsum('ijk,ljk->il', content_features, content_features) / (sum(label.flatten()) / channels - 1)

    # Compute the mean of the style features
    # Multiply each channel by the label to put to zero the non-content features
    style_features = style_features.T # (height * width, cluster size)
    style_mean = np.mean(style_features, axis=(1, ), keepdims=True)
    style_features = style_features - style_mean
    style_covariance = np.einsum('ij,lj->il', style_features, style_features) / (cluster_size - 1)

    content_U, content_S, content_V = np.linalg.svd(content_covariance)
    style_U, style_S, style_V = np.linalg.svd(style_covariance)
    content_D = np.diag(np.power(content_S, -0.5))
    style_D = np.diag(np.power(style_S, 0.5))

    # Compute the whitening and coloring matrix
    whitening_matrix = content_V @ content_D @ content_V.T
    coloring_matrix = style_V @ style_D @ style_V.T

    style_mean = style_mean[:, np.newaxis]
    style_mean = style_mean * label

    result = (coloring_matrix @ whitening_matrix @ content_features.reshape(channels, -1)).reshape(content_features.shape) + style_mean
    result = result * (1 - alpha) + content_features * alpha

    return result