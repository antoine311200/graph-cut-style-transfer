
import networkx as nx
import igraph as ig

import numpy as np
import random
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt




def get_grid_neighbors(i,j,h,w):
    """returns neighbors of pixel (i,j)"""
    neighbors = [(i-1, j), (i+1, j), (i-1, j-1), (i, j-1),(i+1, j-1), (i-1, j+1), (i, j+1), (i+1, j+1)]
    for i, pixel in enumerate(neighbors):
        if pixel[0] < 0 or pixel[1] < 0 or pixel[0] >= h or pixel[1] >= w:
            neighbors[i] = None
    return [node for node in neighbors if node is not None]


def plot_energy(energies, fail_counts=None):
    """plots total energy during iterations"""

    if fail_counts is not None:
        fig, ax = plt.subplots(1,2,figsize = (12,3))

        ax[0].plot(range(len(energies)),energies, marker="x")
        ax[0].set_xlabel("Iterations")
        ax[0].set_ylabel("Total energy")

        ax[1].scatter(range(len(fail_counts)),fail_counts, marker="x")
        ax[1].set_xlabel("Iterations")
        ax[1].set_ylabel("Changes before reduction")

        fig.suptitle("Iterations of alpha expansion")
        fig.tight_layout()
        plt.show();

    else:
        plt.figure(figsize = (4,3))
        plt.plot(range(len(energies)),energies, marker="x")
        plt.xlabel("Iterations")
        plt.ylabel("Total energy")
        plt.show();

def plot_images(target, initial, assigned):
    """shows both images"""

    fig, ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].imshow(target)
    ax[0].axis('off')
    ax[1].imshow(initial)
    ax[1].axis('off')
    ax[2].imshow(assigned)
    ax[2].axis('off')
    
    plt.show();



def alpha_expansion(distances, assignments_, max_cycles=10, beta=1, tolerance=1):#, do_igraph=False):
    _, _, K = distances.shape
    assignments = assignments_.copy()
    energies = []
    total_energies = []
    
    converged = False
    energy_before = np.inf
    cycles = 0
    pbar = tqdm(total=max_cycles*K)
    
    convergence_lag = 0
    while not converged and cycles < max_cycles:
        clusters_list = list(range(K)) #shuffle the order of the clusters
        #random.shuffle(clusters_list)
        
        if convergence_lag==5:
            converged = True

        for alpha in clusters_list:
            # if do_igraph:
            #     min_cut_energy, new_assignments = find_optimal_assignment_ig(distances, assignments, alpha, beta)
            # else:
            min_cut_energy, new_assignments = find_optimal_assignment(distances, assignments, alpha, beta)
            total_energies.append(min_cut_energy)

            if min_cut_energy < energy_before:
                energies.append(min_cut_energy)
                assignments = new_assignments
                energy_before = min_cut_energy
                if converged:
                    converged = np.abs(energy_before - min_cut_energy) <= tolerance #change converged to false if difference big enough
            else:
                convergence_lag += 1

            pbar.update(1)      
        cycles+=1
            
    return assignments, energies, total_energies



def alpha_expansion_graph(distances, assignments, alpha, beta):
    height, width, K = distances.shape

    # graph for min-cut
    G = nx.DiGraph()
    source = f"comp_cluster_{alpha}" #(height, width)
    sink = f"cluster_{alpha}" #(height, width + 1)
    
    #edges between alpha and comp alpha
    for i in range(height):
        for j in range(width):
            if assignments[i,j]==alpha:
                sink_capacity = 0#np.inf
            else:
                sink_capacity = distances[i,j,assignments[i,j]]
            source_capacity = distances[i,j,alpha]
            G.add_edge(source, (i, j), capacity=source_capacity)
            G.add_edge((i, j), sink, capacity=sink_capacity)
            
    #edges between pixels
    for i in range(height):
        for j in range(width):
            k_node = assignments[i,j]
            if k_node != alpha:
                for neighbor in get_grid_neighbors(i,j,height,width):
                    k_neigh = assignments[neighbor[0],neighbor[1]]
                    if k_neigh == alpha or k_neigh == k_node:
                        G.add_edge((i, j), neighbor, capacity = beta)
                    else:
                        new_node = str(sorted([(i,j),neighbor])) # intermediary node between the two nodes
                        if new_node not in G.nodes():
                            G.add_edge(new_node, sink, capacity = beta)
                        G.add_edge((i,j), new_node, capacity = beta)
                        G.add_edge(new_node, (i, j), capacity = 0)#np.inf)

    return G




def find_optimal_assignment(distances, assignments_, alpha, beta):
    height, width, K = distances.shape
    assignments = assignments_.copy()

    # min-cut
    G = alpha_expansion_graph(distances, assignments, alpha, beta)
    cut_value, partition = nx.minimum_cut(G, f"comp_cluster_{alpha}", f"cluster_{alpha}")
    reachable, _ = partition
    
    # update assignments
    for i in range(height):
        for j in range(width):
            if (i, j) in reachable:
                assignments[(i, j)] = alpha
                
    min_cut_energy = compute_energy(distances, assignments, beta)
    
    return min_cut_energy, assignments


def compute_energy(distances, assignments, beta):
    """computes energy of given assigment"""
    height, width, K = distances.shape
    energy = 0
    added_edges = set()
    for i in range(height):
        for j in range(width):
            k = assignments[(i, j)]
            for neighbor in get_grid_neighbors(i,j,height,width):
                if assignments[(i, j)] != assignments[neighbor]:
                    edge_key = tuple(sorted([(i, j), neighbor]))
                    if edge_key not in added_edges:
                        energy += beta
                        added_edges.add(edge_key)
            energy += distances[i, j, k]
    return energy




# def alpha_expansion_graph_ig(distances, assignments_, alpha, beta):
#     height, width, K = distances.shape
#     assignments = assignments_.copy()
    
#     # graph for min-cut
#     G = ig.Graph()
#     source = f"comp_cluster_{alpha}"
#     sink = f"cluster_{alpha}"
#     G.add_vertices([source, sink] + [f"({i},{j})" for i in range(height) for j in range(width)])


#     #edges between alpha and comp alpha
#     for i in range(height):
#         for j in range(width):
#             if assignments[i,j]==alpha:
#                 sink_capacity = np.inf
#             else:
#                 sink_capacity = distances[i,j,assignments[i,j]]
#             source_capacity = distances[i,j,alpha]
#             G.add_edge(source, f"({i},{j})", capacity=source_capacity)
#             G.add_edge(f"({i},{j})", sink, capacity=sink_capacity)
            
#     #edges between pixels
#     for i in range(height):
#         for j in range(width):
#             for neighbor in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
#                 if neighbor[0] >= 0 and neighbor[0] < height and neighbor[1] >= 0 and neighbor[1] < width:

#                     G.add_edge(f"({i},{j})", f"({neighbor[0]},{neighbor[1]})", capacity = beta if assignments[i,j] != assignments[neighbor[0],neighbor[1]] else 0)

#     return G


# def find_optimal_assignment_ig(distances, assignments_, alpha, beta):
#     height, width, K = distances.shape
#     assignments = assignments_.copy()

#     # Compute min-cut
#     G = alpha_expansion_graph_ig(distances, assignments, alpha, beta)
#     partition1, partition2 = G.st_mincut(source=f"comp_cluster_{alpha}", target=f"cluster_{alpha}", capacity='capacity').partition
#     reachable = set(partition1)

#     # Update assignments
#     new_assignments = assignments.copy()
#     for i in range(height):
#         for j in range(width):
#             if f"({i},{j})" in reachable:
#                 new_assignments[(i, j)] = alpha
                
#     min_cut_energy = compute_energy(distances, new_assignments, beta)
    
#     return min_cut_energy, new_assignments


# def compute_energy_ig(distances, assignments, beta):
#     """computes energy of given assigment"""
#     height, width, K = distances.shape
#     energy = 0
#     for i in range(height):
#         for j in range(width):
#             k = assignments[(i, j)]
#             for neighbor in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
#                 if neighbor[0] >= 0 and neighbor[0] < height and neighbor[1] >= 0 and neighbor[1] < width:
#                     if assignments[(i, j)] != assignments[neighbor]:
#                         energy += beta
#                     energy += distances[i, j, k]
#     return energy



def complete_graph(energy_distances):
    """creates grid graph of given width and height and adds edges with cluster centers (weighted by distance)"""
    G = nx.Graph()

    height, width, K = energy_distances.shape

    #pixels
    nodes = [(i, j) for i in range(0, height) for j in range(0, width)]
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



def change_graph(G_, i, j, distances, assignments, new_k, beta):
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







def alpha_expansion_stochastic(distances, assignments_, max_iter=1000, beta=1):
    """imitates alpha expansion energy reduction with random change in pixels"""
    assignments = assignments_.copy()
    height, width, K = distances.shape

    #G = assignment_graph(distances, assignments, beta)
    #energy_before = G.size(weight="weight")
    energy_before = compute_energy(distances, assignments, beta)
    print("Initial energy : ",energy_before)
    energies = [energy_before] # decreased energies
    computed_energies = [energy_before] #all computations 
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
            computed_energies.append(new_energy)
        else:
            fail_counts[-1] += 1

    print("Final energy : ",energy_before)

    return assignments, energies, fail_counts, computed_energies


def alpha_expansion_greedy(distances, assignments_, max_iter=1000, beta=1):
    """imitates alpha expansion energy reduction with greedy change in pixels"""
    assignments = assignments_.copy()
    height, width, K = distances.shape
    energy_before = compute_energy(distances, assignments, beta)
    print("Initial energy : ",energy_before)
    energies = [energy_before] # decreased energies
    computed_energies = [energy_before] #all computations 
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
        new_energy = new_energies[k]
        computed_energies.append(new_energy)

        if k != current_k:
            if new_energy <= energy_before:
                energy_before = new_energy
                assignments[i,j] = k
                energies.append(new_energy)
                fail_counts.append(0)
            else:
                fail_counts[-1] += 1
        else:
            fail_counts[-1] += 1
    print("Final energy : ",energy_before)
    return assignments, energies, fail_counts, computed_energies