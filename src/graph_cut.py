
import networkx as nx
import igraph as ig

import numpy as np
import random
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt



def get_grid_neighbors(i,j,h,w,total=False):
    """returns neighbors of pixel (i,j)"""
    if total: #all neighbors
        neighbors = [(i-1, j), (i+1, j), (i-1, j-1), (i, j-1),(i+1, j-1), (i-1, j+1), (i, j+1), (i+1, j+1)]
    else: #only left/right top/bottom
        neighbors = [(i-1, j), (i+1, j), (i, j-1),(i, j+1)]
    for i, pixel in enumerate(neighbors):
        if pixel[0] < 0 or pixel[1] < 0 or pixel[0] >= h or pixel[1] >= w:
            neighbors[i] = None
    return [node for node in neighbors if node is not None]


def plot_energy(energies, total_energies, fail_counts=None):
    """plots energy reduction and total energy during iterations, can add fail_counts"""

    if fail_counts:
        fig, ax = plt.subplots(1,3,figsize = (12,3))
    else:
        fig, ax = plt.subplots(1,2,figsize = (8,3))

    ax[0].plot(range(len(total_energies)),total_energies, marker="x")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Total energy")

    ax[1].plot(range(len(energies)),energies, marker="x")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Energy reduction")

    if fail_counts:
        ax[2].scatter(range(len(fail_counts)),fail_counts, marker="x")
        ax[2].set_xlabel("Iterations")
        ax[2].set_ylabel("Changes before reduction")

    fig.suptitle("Iterations of alpha expansion")
    fig.tight_layout()
    plt.show();



def alpha_expansion_stochastic(distances, assignments_, max_iter=1000, lambd=1):
    """imitates alpha expansion energy reduction with random change in pixels"""
    assignments = assignments_.copy()
    height, width, K = distances.shape

    #G = assignment_graph(distances, assignments, lambd)
    #energy_before = G.size(weight="weight")
    energy_before = compute_energy(distances, assignments, lambd)
    print("Initial energy : ",energy_before)
    energies = [energy_before] # decreased energies
    computed_energies = [energy_before] #all computations 
    fail_counts = [0]

    for iteration in tqdm(range(max_iter)):
        i,j  = random.randint(0, height-1), random.randint(0, width-1)
        current_k = assignments[(i,j)]
        k = random.randint(0,K-1)
        
        if k != current_k:
            #new_G = change_graph(G, i, j, distances, assignments, current_k, k, lambd)
            #new_energy = new_G.size(weight="weight")
            new_assignments = assignments.copy()
            new_assignments[i,j] = k
            new_energy = compute_energy(distances, new_assignments, lambd)
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


def alpha_expansion_greedy(distances, assignments_, max_iter=1000, lambd=1):
    """imitates alpha expansion energy reduction with greedy change in pixels"""
    assignments = assignments_.copy()
    height, width, K = distances.shape
    energy_before = compute_energy(distances, assignments, lambd)
    print("Initial energy : ",energy_before)
    energies = [energy_before] # decreased energies
    computed_energies = [energy_before] #all computations 
    fail_counts = [0]
    
    for iteration in tqdm(range(max_iter)):
        i,j  = random.randint(0, height-1), random.randint(0, width-1)
        current_k = assignments[i,j]

        new_energies = []
        for k in range(K):
            if k != current_k:
                new_assignments = assignments.copy()
                new_assignments[i,j] = k
                new_energies.append(compute_energy(distances, new_assignments, lambd))
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


from networkx.algorithms.flow import shortest_augmenting_path

def alpha_expansion(distances, assignments_, max_cycles=10, lambd=1, tolerance=1, method="default"):
    """computes alpha expansion and returns optimal assigmnent"""
    _, _, K = distances.shape
    assignments = assignments_.copy()    
    energy_before = compute_energy(distances, assignments, lambd)
    energies = [energy_before] # decreased energies
    computed_energies = [energy_before] #all computations 
    cycles = 0
    pbar = tqdm(total=max_cycles*K)

    converged = False
    while not converged and cycles < max_cycles:
        clusters_list = list(range(K)) 
        random.shuffle(clusters_list) #shuffle the order of the clusters to avoid bias
        
        converged = True #stops unless finds energy reduction
        for alpha in clusters_list:
            min_cut_energy, new_assignments = find_optimal_assignment(distances, assignments, alpha, lambd, method=method)
            computed_energies.append(min_cut_energy)

            if min_cut_energy < energy_before:
                energies.append(min_cut_energy)
                assignments = new_assignments
                energy_before = min_cut_energy
                if converged:
                    converged = np.abs(energy_before - min_cut_energy) <= tolerance #change converged to false if difference big enough

            pbar.update(1)      
        cycles+=1
            
    return assignments, energies, computed_energies



def alpha_expansion_graph(distances, assignments, alpha, lambd):
    height, width, K = distances.shape

    G = nx.DiGraph()
    source = f"comp_cluster_{alpha}"
    sink = f"cluster_{alpha}"
    
    #edges between alpha and comp alpha
    for i in range(height):
        for j in range(width):
            source_capacity = distances[i,j,alpha]
            G.add_edge(source, (i, j), capacity=source_capacity)
            if assignments[i,j]==alpha:
                G.add_edge((i, j), sink)#, capacity = float("inf"))
            else:
                sink_capacity = distances[i,j,assignments[i,j]]
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
                        G.add_edge((i, j), neighbor, capacity = lambd)
                    else:
                        new_node = str(sorted([(i,j),neighbor])) # intermediary node between the two nodes
                        if new_node not in G.nodes():
                            G.add_edge(new_node, sink, capacity = lambd)
                        G.add_edge((i,j), new_node, capacity = lambd)
                        G.add_edge(new_node, (i, j))#, capacity = float("inf"))

    return G



def alpha_expansion_graph_ig(distances, assignments, alpha, lambd):
    height, width, K = distances.shape

    # graph for min-cut
    G = ig.Graph(directed=True)
    source = f"comp_cluster_{alpha}"
    sink = f"cluster_{alpha}"
    G.add_vertices([source, sink] + [f"({i},{j})" for i in range(height) for j in range(width)])

    
    #edges between alpha and comp alpha
    for i in range(height):
        for j in range(width):
            source_capacity = distances[i,j,alpha]
            G.add_edge(source, f"({i},{j})", capacity=source_capacity)
            if assignments[i,j]==alpha:
                G.add_edge(f"({i},{j})", sink)#, capacity = float("inf"))
            else:
                sink_capacity = distances[i,j,assignments[i,j]]
                G.add_edge(source, f"({i},{j})", capacity=source_capacity)
                G.add_edge(f"({i},{j})", sink, capacity=sink_capacity)
            
    #edges between pixels
    for i in range(height):
        for j in range(width):
            k_node = assignments[i,j]
            if k_node != alpha:
                for neighbor in get_grid_neighbors(i,j,height,width):
                    k_neigh = assignments[neighbor[0],neighbor[1]]
                    if k_neigh == alpha or k_neigh == k_node:
                        G.add_edge(f"({i},{j})", f"({neighbor[0]},{neighbor[1]})", capacity = lambd)
                    else:
                        new_node = str(sorted([f"({i},{j})",f"({neighbor[0]},{neighbor[1]})"])) # intermediary node between the two nodes
                        if new_node not in G.vs:
                            G.add_vertices([new_node])
                            G.add_edge(new_node, sink, capacity = lambd)
                        G.add_edge(f"({i},{j})", new_node, capacity = lambd)
                        G.add_edge(new_node, f"({i},{j})")#, capacity = float("inf"))
    return G




def find_optimal_assignment(distances, assignments_, alpha, lambd, method="default"):
    height, width, K = distances.shape
    assignments = assignments_.copy()

    # min-cut
    if method=="igraph":
        G = alpha_expansion_graph_ig(distances, assignments, alpha, lambd)
        _, reachable = G.st_mincut(source=f"comp_cluster_{alpha}", target=f"cluster_{alpha}", capacity='capacity').partition
    else:
        G = alpha_expansion_graph(distances, assignments, alpha, lambd)
        if method=="shortest_path":
            cut_value, partition = nx.minimum_cut(G, f"comp_cluster_{alpha}", f"cluster_{alpha}",flow_func=shortest_augmenting_path)
        else:
            cut_value, partition = nx.minimum_cut(G, f"comp_cluster_{alpha}", f"cluster_{alpha}")
        _, reachable = partition
    
    # update assignments
    for i in range(height):
        for j in range(width):
            if (i,j) in reachable:
                assignments[i,j] = alpha
                
    min_cut_energy = compute_energy(distances, assignments, lambd)
    
    return min_cut_energy, assignments


def compute_energy(distances, assignments, lambd):
    """computes energy of given assigment"""
    height, width, K = distances.shape
    energy = 0
    added_edges = set()
    for i in range(height):
        for j in range(width):
            k = assignments[(i, j)]
            energy += distances[i, j, k]
            for neighbor in get_grid_neighbors(i,j,height,width):
                edge_key = tuple(sorted([(i, j), neighbor]))
                if edge_key not in added_edges:
                    added_edges.add(edge_key)
                    if  assignments[neighbor[0],neighbor[1]] != k:
                        energy += lambd
    return energy






## not used

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


def assignment_graph(energy_distances,assignments, lambd):
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
                        G.add_edge((i,j),neighbor,weight=lambd)
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



def change_graph(G_, i, j, distances, assignments, new_k, lambd):
    """updates assignment graph edges based on cluster change"""
    G = G_.copy()
    current_neighbors = list(G.neighbors((i,j)))

    for neighbor in get_grid_neighbors(i,j):
        if neighbor in current_neighbors:
            G.remove_edge((i,j), neighbor)
        else:
            k2 = assignments[neighbor[0], neighbor[1]]
            if new_k != k2:
                G.add_edge((i,j),neighbor,weight=lambd)
    G.add_edge((i,j),f"cluster_{new_k}", weight=distances[i,j,new_k])
    
    return G