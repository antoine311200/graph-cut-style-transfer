
import networkx as nx
import numpy as np
import random
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

def complete_graph(energy_distances):
    """creates grid graph of given width and height and adds edges with cluster centers (weighted by distance)"""
    G = nx.Graph()

    height, width, K = energy_distances.shape

    #pixels
    nodes = [(i, j) for i in range(0, w) for j in range(0, h)]
    G.add_nodes_from(nodes)

    #horizontal and vertical edges
    edges_h = [((i, j), (i + 1, j)) for i in range(0, height - 1) for j in range(0, width)]
    edges_v = [((i, j), (i, j + 1)) for i in range(0, height) for j in range(0, width - 1)]
    G.add_edges_from(edges_h)
    G.add_edges_from(edges_v)

    #cluster nodes
    cluster_nodes = [f'cluster_{k}' for k in range(K)]
    G.add_nodes_from(cluster_nodes)

    #cluster edges
    for i in range(height):
        for j in range(width):
            for k in range(K):
                cluster_node = cluster_nodes[k]
                weight = energy_distances[i, j, k]
                G.add_edge((i, j), cluster_node, weight=weight)

    return G



def assignment_graph(energy_distances,assignments, beta):
    """creates graph with edges representing energy (distance to assigned cluster and neighbors with different clusters)"""
    G = nx.Graph()

    height, width, K = energy_distances.shape

    #pixels
    nodes = [(i, j) for i in range(0, height) for j in range(0, width)]
    G.add_nodes_from(nodes)

    added_edges = set()
    # energy edges of nodes belonging to different clusters
    for node in G.nodes():
        if "cluster" not in node:
            i,j = node[0], node [1]
            for neighbor in get_grid_neighbors(i,j):
                if ((i,j),neighbor) not in added_edges:        
                    k1, k2 = assignments[i, j], assignments[neighbor[0], neighbor[1]]
                    if k1 != k2:
                        G.add_edge((i,j),neighbor,weight=beta)
                    added_edges.add(((i,j),neighbor))

    #cluster nodes
    cluster_nodes = [f'cluster_{k}' for k in range(K)]
    G.add_nodes_from(cluster_nodes)

    #cluster distance edges
    for node in G.nodes():
        if "cluster" not in node:
            i,j = node[0], node[1]
            k = assignments[i, j]
            weight = energy_distances[i, j, k]
            G.add_edge((i, j), f'cluster_{k}', weight=weight)

    return G



def get_grid_neighbors(i,j,h=10,w=10):
    """returns neighbors of pixel (i,j)"""
    neighbors = [(i-1, j), (i+1, j), (i-1, j-1), (i, j-1),(i+1, j-1), (i-1, j+1), (i, j+1), (i+1, j+1)]
    for i, pixel in enumerate(neighbors):
        if pixel[0] < 0 or pixel[1] < 0 or pixel[0] >= h or pixel[1] >= w:
            neighbors[i] = None
    return [node for node in neighbors if node is not None]


def change_graph(G_, i, j, distances, assignments, old_k, new_k, beta):
    """updates assignment graph edges based on cluster change"""
    G = G_.copy()
    current_neighbors = list(G.neighbors((i,j)))

    for neighbor in get_grid_neighbors(i,j):
        if neighbor in current_neighbors:
            G.remove_edge((i,j), neighbor)
        else:
            k2 = assignments[neighbor[0], neighbor[1]]
            if new_k != k2:
                G.add_edge((i,j),neighbor,weight=beta)
    G.add_edge((i,j),f"cluster_{new_k}", weight=distances[i,j,new_k])
    
    return G




def alpha_expansion_stochastic(distances, assignments, max_iter=100, beta=1):
    #G = assignment_graph(distances, assignments, beta)
    #energy_before = G.size(weight="weight") 
    energy_before = compute_energy(distances, assignments, beta)
    energies = [energy_before]
    height, width, K = distances.shape

    fail_counts = [0]
    for iteration in tqdm(range(max_iter)):
        i,j  = random.randint(0, height-1), random.randint(0, width-1)
        current_k = assignments[(i,j)]
        k = random.randint(0,K-1)
        
        if k != current_k:
            #new_G = change_graph(G, i, j, distances, assignments, current_k, k, beta)
            #new_energy = new_G.size(weight="weight")
            new_assignments = assignments.copy()
            new_assignments[i,j] = k
            new_energy = compute_energy(distances, new_assignments, beta)
            if new_energy <= energy_before:
                energy_before = new_energy
                assignments[i,j] = k
                energies.append(new_energy)
                fail_counts.append(0)
            else:
                fail_counts[-1] += 1

        else:
            fail_counts[-1] += 1
            
    return assignments, energies, fail_counts


def alpha_expansion_greedy(distances, assignments, max_iter=100, beta=1):
    #G = assignment_graph(distances, assignments, beta)
    #energy_before = G.size(weight="weight")
    energy_before = compute_energy(distances, assignments, beta)
    energies = [energy_before]
    height, width, K = distances.shape

    fail_counts = [0]
    for iteration in tqdm(range(max_iter)):
        i,j  = random.randint(0, height-1), random.randint(0, width-1)
        current_k = assignments[i,j]

        new_energies = []
        for k in range(K):
            if k!= current_k:
                new_assignments = assignments.copy()
                new_assignments[i,j] = k
                new_energies.append(compute_energy(distances, new_assignments, beta))
            else:
                new_energies.append(energy_before)
        k = np.argmin(new_energies)
        
        if k != current_k:
            #new_G = change_graph(G, i, j, distances, assignments, current_k, k, beta)
            #new_energy = new_G.size(weight="weight")
            new_energy = new_energies[k]
            if new_energy <= energy_before:
                energy_before = new_energy
                assignments[i,j] = k
                energies.append(new_energy)
                fail_counts.append(0)
            else:
                fail_counts[-1] += 1
        else:
            fail_counts[-1] += 1
            
    return assignments, energies, fail_counts


def plot_energy(energies, fail_counts):
    """plots total energy during iterations"""
    fig, ax = plt.subplots(1,2,figsize = (10,3))

    ax[0].plot(range(len(energies)),energies, marker="x")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Total energy")

    ax[1].plot(range(len(fail_counts)),fail_counts, marker="x")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Changes before reduction")

    fig.suptitle("Iterations of alpha expansion")
    fig.tight_layout()
    plt.show();



def alpha_expansion(distances, assignments, max_iter=100, beta=1):
    height, width, K = distances.shape
    energies = []
    fail_counts = []
    
    energy_before = np.inf
    for iteration in tqdm(range(max_iter)):
        k = random.randint(0, K - 1) #random label to test new assignments on
        min_cut_energy, new_assignments = find_optimal_assignment(distances, assignments, k, beta)
        
        if min_cut_energy < energy_before:
            assignments = new_assignments
            energies.append(min_cut_energy)
            fail_counts.append(0)
        else:
            fail_counts[-1] += 1
            
    return assignments, energies, fail_counts



def alpha_expansion_graph(distances, assignments, alpha, beta):
    height, width, K = distances.shape
    
    # Construct graph for min-cut
    G = nx.Graph()
    source = f"comp_cluster_{alpha}" #(height, width)
    sink = f"cluster_{alpha}" #(height, width + 1)
    
    #edges between alpha and comp alpha
    for i in range(height):
        for j in range(width):
            if assignments[i,j]==alpha:
                sink_capacity = np.inf
            else:
                sink_capacity = distances[i,j,alpha]
            source_capacity = distances[i,j,assignments[i,j]]
            G.add_edge(source, (i, j), capacity=source_capacity)
            G.add_edge((i, j), sink, capacity=sink_capacity)
            
    #edges between pixels
    for i in range(height):
        for j in range(width):
            for neighbor in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                if neighbor[0] >= 0 and neighbor[0] < height and neighbor[1] >= 0 and neighbor[1] < width:
                    G.add_edge((i, j), neighbor, capacity = beta if assignments[i,j] != assignments[neighbor[0],neighbor[1]] else 0)

    return G


def find_optimal_assignment(distances, assignments, alpha, beta):
    height, width, K = distances.shape

    # Compute min-cut
    G = alpha_expansion_graph(distances, assignments, alpha, beta)
    cut_value, partition = nx.minimum_cut(G, f"comp_cluster_{alpha}", f"cluster_{alpha}")
    reachable, _ = partition
    
    # Update assignments
    new_assignments = assignments.copy()
    for i in range(height):
        for j in range(width):
            if (i, j) in reachable:
                new_assignments[(i, j)] = alpha
                
    min_cut_energy = compute_energy(distances, new_assignments, beta)
    
    return min_cut_energy, new_assignments


def compute_energy(distances, assignments, beta):
    """computes energy of given assigment"""
    height, width, K = distances.shape
    energy = 0
    for i in range(height):
        for j in range(width):
            k = assignments[(i, j)]
            for neighbor in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                if neighbor[0] >= 0 and neighbor[0] < height and neighbor[1] >= 0 and neighbor[1] < width:
                    if assignments[(i, j)] != assignments[neighbor]:
                        energy += beta
                    energy += distances[i, j, k]
    return energy
