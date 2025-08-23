import random
import time
import cv2
import dgl
import numpy as np
import torch
from collections import deque
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
from scipy.spatial import cKDTree, Delaunay
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity


class Timer:
    """
    A timing context manager that measures the execution time of a code block. It can measure time in seconds or nanoseconds according to user needs. 
        use_ns (bool): If True, the timer measures time in nanoseconds; if False, the unit is seconds. 
        start (float|int): The time to start the measurement. 
        end (float|int): The time at which the measurement ends. 
        interval (float|int): The duration between the calculated start and end times.
    """

    def __init__(self, note='', use_ns=False):
        """
        Use to select whether to initialize the Timer with nanosecond precision. 
        parameter:
            use_ns (bool): Determines whether to use nanoseconds for time measurement, default is False.
        """
        self.use_ns = use_ns
        self.start = None
        self.end = None
        self.interval = None
        self.note = note

    def __enter__(self):
        """
        Start the timer. Record the start time when entering the context block. 
        return:
            Timer: Returns its own object to access properties outside the context.
        """
        self.start = time.perf_counter_ns() if self.use_ns else time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        End timer. Record the end time when the context block is exited. 
        This function also calculates the time interval and prints the elapsed time.

        parameter:
            excc_type: If an exception is raised in the context, it is an exception type. 
            exc_value: If an exception is raised, it is an outlier. 
            traceback: If an exception occurs, it is the traceback details. 
        return:
            None
        """
        self.end = time.perf_counter_ns() if self.use_ns else time.perf_counter()
        self.interval = self.end - self.start
        print((f'{self.note} ' if self.note else '') + f"耗时：{self.interval:.6f} 秒")


#-------------------------------------------------------------#

def build_dgl_graph_supernode(super_nodes, device):
    # Assume that super_nodes is a list of supernodes for a single graph
    g = dgl.DGLGraph()
    g.add_nodes(len(super_nodes))

    # Add node features
    node_features = torch.stack([node['descriptor'] for node in super_nodes]).to(device)
    g.ndata['feat'] = node_features

    # Add edges
    src, dst = [], []
    for super_node in super_nodes:
        for edge_target in super_node['edges']:
            src.append(super_node['id'])
            dst.append(edge_target)
    g.add_edges(src, dst)

    return g

def create_supernodes_batched2(batched_graph, n, strategy='max_neighbors', exclude_used_nodes=True):
    batched_supernodes = []

    for graph in batched_graph:
        supernodes = []
        supernode_edges = {}
        visited = set() if exclude_used_nodes else None

        def local_search(start_node_index):
            if visited is not None and start_node_index in visited:
                return None, None
            supernode_id = len(supernodes)
            queue = deque([start_node_index])
            supernode = {'id': supernode_id, 'nodes': [start_node_index], 'edges': []}
            if visited is not None:
                visited.add(start_node_index)

            descriptors = [graph[start_node_index]['descriptor']]
            while queue and len(supernode['nodes']) < n:
                current_index = queue.popleft()
                for neighbor_index in graph[current_index]['edges']:
                    if visited is None or neighbor_index not in visited:
                        supernode['nodes'].append(neighbor_index)
                        queue.append(neighbor_index)
                        if visited is not None:
                            visited.add(neighbor_index)
                        descriptors.append(graph[neighbor_index]['descriptor'])
                        for sn_id, sn_nodes in supernode_edges.items():
                            if neighbor_index in sn_nodes and supernode_id != sn_id:
                                supernode['edges'].append(sn_id)
                                break

            # Calculate average scores and descriptors using PyTorch
            scores_tensor = torch.stack([graph[node]['score'].clone().detach() for node in supernode['nodes']])
            supernode['score'] = scores_tensor.mean().item()
            # Directly use the coordinates of the starting node as the center coordinates of the super node
            supernode['center'] = graph[start_node_index]['point'].clone().detach().cpu().numpy()

            if descriptors:
                supernode['descriptor'] = torch.stack(descriptors).mean(dim=0).cpu().numpy()
            else:
                supernode['descriptor'] = torch.zeros(128, device=graph[0]['descriptor'].device).cpu().numpy()
            supernode_edges[supernode_id] = supernode['nodes']
            return supernode, supernode_edges

        for i in range(len(graph)):
            if visited is None or i not in visited:
                new_supernode, _ = local_search(i)
                if new_supernode:
                    supernodes.append(new_supernode)
        for supernode in supernodes:
            supernode['edges'] = list(set(supernode['edges']))
        batched_supernodes.append(supernodes)
    return batched_supernodes

# Cluster and generate supernodes
def create_supernodes_batched_cluster(batched_graph, n_clusters=None):
    # Process each layer and generate supernodes
    super_graphs = []

    for graph in batched_graph:
        descriptors = np.array([node['descriptor'].cpu().numpy() for node in graph])
        n_clusters = n_clusters or int(len(graph) / 4)
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit(descriptors)
        labels = kmeans.labels_
        super_nodes = []
        node_id_to_super_node = {}

        for cluster_id in range(n_clusters):
            cluster_nodes = [node for node, label in zip(graph, labels) if label == cluster_id]
            if not cluster_nodes:
                continue
            # Select the node with the highest score as the center
            center_node = max(cluster_nodes, key=lambda node: node['score'])
            avg_score = torch.stack([node['score'] for node in cluster_nodes]).mean()
            avg_descriptor = torch.stack([node['descriptor'] for node in cluster_nodes]).mean(dim=0)

            super_node = {
                'id': cluster_id,
                'nodes': [node['id'] for node in cluster_nodes],
                'edges': [],  # Will be filled in next steps
                'score': avg_score,
                'center': center_node['point'],
                'descriptor': avg_descriptor,
            }

            for node in cluster_nodes:
                node_id_to_super_node[node['id']] = super_node['id']
            super_nodes.append(super_node)

        # Handle edges between supernodes
        for super_node in super_nodes:
            edge_super_nodes = set()
            for node_id in super_node['nodes']:
                original_node = next((node for node in graph if node['id'] == node_id), None)
                if original_node:
                    for edge in original_node['edges']:
                        edge_super_node_id = node_id_to_super_node.get(edge)
                        if edge_super_node_id and edge_super_node_id != super_node['id']:
                            edge_super_nodes.add(edge_super_node_id)
            super_node['edges'] = list(edge_super_nodes)
        super_graphs.append(super_nodes)
    return super_graphs

def create_supernodes_batched(batched_graph, radius):
    super_graphs = []

    for graph in batched_graph:
        points = np.array([node['point'].cpu().numpy() for node in graph])
        # Calculate the Euclidean distance between all nodes
        distances = euclidean_distances(points, points)
        # Determine which nodes should be connected based on the radius
        adjacency = distances < radius

        super_nodes = []
        node_id_to_super_node = {}

        for idx, node in enumerate(graph):
            connected_nodes = np.where(adjacency[idx])[0]
            if len(connected_nodes) > 0:
                avg_descriptor = torch.stack([graph[i]['descriptor'] for i in connected_nodes]).mean(dim=0)

                super_node = {
                    'id': idx,
                    'nodes': connected_nodes.tolist(),
                    'edges': [],
                    'score': node['score'],
                    'point': node['point'],
                    'descriptor': avg_descriptor,
                }

                for connected_node_id in connected_nodes:
                    node_id_to_super_node[connected_node_id] = idx
                super_nodes.append(super_node)

        # Handle edges between supernodes
        for super_node in super_nodes:
            edge_super_nodes = set()
            for node_id in super_node['nodes']:
                original_node = graph[node_id]
                for edge in original_node['edges']:
                    edge_super_node_id = node_id_to_super_node.get(edge)
                    if edge_super_node_id and edge_super_node_id != super_node['id']:
                        edge_super_nodes.add(edge_super_node_id)
            super_node['edges'] = list(edge_super_nodes)

        super_graphs.append(super_nodes)
    return super_graphs

def process_batched_supernodes(supernodes_batched, device):
    keypoints_list = []
    descriptors_list = []
    scores_list = []
    for supernodes in supernodes_batched:
        keypoints_tensor = torch.stack([s['point'] for s in supernodes]).float()
        descriptors_tensor = torch.stack([s['descriptor'] for s in supernodes]).float()
        scores_tensor = torch.stack([s['score'] for s in supernodes]).float()
        keypoints_list.append(keypoints_tensor)
        descriptors_list.append(descriptors_tensor)
        scores_list.append(scores_tensor)

    keypoints_batched = torch.stack(keypoints_list).to(device)
    descriptors_batched = torch.stack(descriptors_list).to(device)
    scores_batched = torch.stack(scores_list).to(device)
    descriptors_batched = descriptors_batched.permute(0, 2, 1)
    return keypoints_batched, descriptors_batched, scores_batched

#-------------------------------------------------------------#

def visualize_graph_with_coordinates(graph,
                                     show_node_scores=False,
                                     node_size_range=(3, 300),
                                     edge_alpha=0.8,
                                     edge_width=1,
                                     figsize=(8, 8),
                                     title="Graph Visualization",
                                     image=None):
    """
    Visualize NetworkX graphs with point coordinates (node ​​positions are obtained from node["point"]).

    parameter:
        graph (nx.Graph): Graph containing point/feat/score attributes 
        show_node_scores (bool): Whether to map node colors with scores 
        node_size_range (tuple): Minimum and maximum node size 
        edge_alpha (float): edge transparency 
        figsize (tuple): image size 
        title (str): title 
        image (np.ndarray): if provided as image background (HxW or HxWx3)
    """
    pos = {i: data['point'][:2] for i, data in graph.nodes(data=True)}  # Only take the first two dimensions (xy)
    scores = np.array([data['score'] for _, data in graph.nodes(data=True)])

    # Node color and size
    if show_node_scores:
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        node_sizes = norm_scores * (node_size_range[1] - node_size_range[0]) + node_size_range[0]
        node_colors = norm_scores
    else:
        node_sizes = node_size_range[0]
        node_colors = "blue"
    plt.figure(figsize=figsize)
    # Draw the background image (if any)
    if image is not None:
        if image.ndim == 2:  # Grayscale
            plt.imshow(image, cmap='gray', origin='upper')
        else:  # Color
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), origin='upper')
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color="red",
        width=edge_width,
        alpha=edge_alpha,
        cmap=plt.cm.viridis,
    )
    plt.title(title)
    plt.axis("equal")
    plt.show()

def create_graph_from_keypoints(keypoints, descriptors, scores, radius, device=torch.device('cpu')):
    def calc_distances_numpy(points):
        return np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2))
    def calc_distances_torch(points):
        return torch.sqrt(torch.sum((points.unsqueeze(1) - points.unsqueeze(0)) ** 2, dim=2))
    batched_graph = []
    for batched_idx in range(keypoints.shape[0]):
        kpts, descs, scrs = keypoints[batched_idx], descriptors[batched_idx].permute(1, 0), scores[batched_idx]
        dist_matrix = calc_distances_numpy(kpts) if isinstance(kpts, np.ndarray) else calc_distances_torch(kpts).cpu().numpy()
        edges = dist_matrix < radius
        graph = []
        for i in range(kpts.shape[0]):
            kpt, desc, scr = kpts[i], descs[i], scrs[i]
            connected_nodes = np.where(edges[i])[0].tolist()
            # Ensure there are no isolated nodes
            if len(connected_nodes) == 1:  # Only connected to itself
                # Find the nearest node (excluding itself) and connect
                nearest_node = np.argsort(dist_matrix[i])[1]  # [0] would be the node itself
                connected_nodes.append(nearest_node)

            graph.append({
                'id': i,
                'point': kpt,
                'score': scr,
                'descriptor': desc,
                'edges': connected_nodes,
            })

        batched_graph.append(graph)
        # batched_graph.append(remove_small_components(graph))
    return batched_graph

def create_dgl_graph_from_batched_graph(batched_graph, device=torch.device('cpu')):
    dgl_graphs = []

    for graph in batched_graph:
        # Create an empty DGL diagram
        g = dgl.graph(([], []), num_nodes=len(graph), device=device)

        # Extract and set node features
        g.ndata['feat'] = torch.stack([node['descriptor'] for node in graph]).to(device)
        g.ndata['point'] = torch.stack([node['point'] for node in graph]).to(device)
        g.ndata['score'] = torch.stack([node['score'] for node in graph]).to(device)

        # Extract and add edges
        edges_src = []
        edges_dst = []
        for node in graph:
            for neighbor in node['edges']:
                edges_src.append(node['id'])
                edges_dst.append(neighbor)
        src_tensor = torch.tensor(edges_src, dtype=torch.long, device=device)
        dst_tensor = torch.tensor(edges_dst, dtype=torch.long, device=device)
        g.add_edges(src_tensor, dst_tensor)

        # Save the built DGL graph
        dgl_graphs.append(g)

    return dgl_graphs

def fast_percentile_threshold(similarities, percentile):
    """
    Fast approximate quantile calculations
    For large-scale data, this method is faster than np.percentile, and is especially suitable for quantile filtering of edge similarity in graph construction.

    parameter:
        similarities (np.ndarray or torch.Tensor): Similarity vector (one-dimensional). 
        percentile (float): Quantile (0~100), such as 50 represents the median. 
    return:
        float: Approximate quantile values.
    """
    k = int(len(similarities) * percentile / 100)
    if k >= len(similarities): k = len(similarities) - 1
    return np.partition(similarities, k)[k]

def fast_cosine_similarity_matrix(descs: np.ndarray) -> np.ndarray:
    """
    Quickly calculate the cosine similarity matrix using PyTorch. 
    Input: descs: np.ndarray of shape (N, D), unnormalized eigenvector 
    Output: sim_matrix: np.ndarray of shape (N, N), normalized cosine similarity matrix
    """
    descs = torch.from_numpy(descs).float()  # Convert to float32 tensor
    descs = torch.nn.functional.normalize(descs, dim=1)  # L2 Normalization
    sim_matrix = torch.matmul(descs, descs.T)
    return sim_matrix.cpu().numpy()

def calculate_cosine_similarity_threshold(descriptors, percentile=50, similarity_matrix=None):
    """
    Based on the pairwise cosine similarity of a batch of descriptors, the dynamic threshold for the specified percentile is calculated. 
    To avoid constructing overly dense graphs, this function can adaptively calculate a threshold of cosine similarity, only edges above which are retained for subsequent graph construction.
    parameter:
        descriptors (np.ndarray): shape=(N, D), descriptor vector for key points (not normalized). 
        percentile (float): Quantile used to set the threshold (for example, 50 means the median). 
        Similarity_matrix (np.ndarray|None): Optional, pre-calculated similarity matrix, if None, it will be calculated automatically. 
    return:
        float: The similarity threshold calculated based on the specified percentage.
    """
    # similarity_matrix = cosine_similarity(descriptors) if similarity_matrix is None else similarity_matrix
    similarity_matrix = fast_cosine_similarity_matrix(descriptors) if similarity_matrix is None else similarity_matrix
    similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    similarities = similarities[similarities > 0]
    if len(similarities) == 0: return 0.0  # fallback value to avoid crashes
    # similarity_threshold = np.percentile(similarities, percentile)
    similarity_threshold = fast_percentile_threshold(similarities, percentile)
    return similarity_threshold

def fast_build_graph_with_cosine_similarity(keypoints, descriptors, scores, radius, percentile=50):
    """
    Construct a spatial map with cosine similarity filtering (supports dynamic threshold, fast similarity calculation, KDTree batch adjacency).

    parameter:
        keypoints (torch.Tensor): shape (B, N, 2 or 3), keypoint coordinates per batch 
        descriptors (torch.Tensor): shape (B, D, N), keypoint descriptors per batch (unnormalized) 
        scores (torch.Tensor): shape (B, N), keypoint scores per batch 
        radius (float): Adjacent radius threshold 
        percentile (float): Cosine similarity percentile threshold for filtering 
    return:
        List[nx.Graph]: NetworkX expression form corresponding to each batch of graphs
    """
    batch_graphs = []
    B = keypoints.shape[0]
    for b in range(B):
        # === Step 1: Extract node data from each graph ===
        kpts = keypoints[b].cpu().numpy()  # shape (N, 2/3)
        descs = descriptors[b].permute(1, 0).cpu().numpy()  # shape (N, D)
        scrs = scores[b].cpu().numpy()  # shape (N,)
        N = len(kpts)
        # === Step 2: Create KDTree and get all point pairs (i,j) in radius, i<j ===
        kdtree = cKDTree(kpts)
        edge_candidates = kdtree.query_pairs(r=radius)  # set of (i, j)
        # === Step 3: Calculate the cosine similarity matrix ===
        sim_matrix = fast_cosine_similarity_matrix(descs)  # (N, N)
        similarity_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        threshold = fast_percentile_threshold(similarity_values, percentile)
        # === Step 4: Build a graph ===
        graph = nx.Graph()
        for i in range(N):
            graph.add_node(i, point=kpts[i], feat=descs[i], score=scrs[i])
        for i, j in edge_candidates:
            if sim_matrix[i, j] >= threshold:
                graph.add_edge(i, j)
        batch_graphs.append(graph)
    return batch_graphs

def build_graph_with_cosine_similarity(keypoints, descriptors, scores, radius, percentile=50):
    keypoints = keypoints.cpu().numpy()
    descriptors = descriptors.permute(0, 2, 1).cpu().numpy() # => (batch, number, dim)
    scores = scores.cpu().numpy()
    batch_graph = []
    for batched_idx in range(keypoints.shape[0]):
        kpts = keypoints[batched_idx]
        descs = descriptors[batched_idx]
        scrs = scores[batched_idx]
        kdtree = cKDTree(kpts)
        graph = nx.Graph()
        # similarity_matrix = cosine_similarity(descs)
        similarity_matrix = fast_cosine_similarity_matrix(descs)
        similarity_threshold = calculate_cosine_similarity_threshold(descs, percentile)
        # Create a node
        for i in range(len(kpts)): graph.add_node(i, point=kpts[i], feat=descs[i], score=scrs[i])
        # Add edges by space and descriptor similarity
        for i, kpt in enumerate(kpts):
            neighbors_idx = kdtree.query_ball_point(kpt, r=radius)
            for j in neighbors_idx:
                if i != j and similarity_matrix[i, j] >= similarity_threshold:
                    graph.add_edge(i, j)
        batch_graph.append(graph)
    return batch_graph

def connect_isolated_nodes(graph):
    """
    Connect isolated nodes in the graph to their nearest neighbor, using positions stored in node attributes.

    Parameters:
    - graph: networkx.Graph, the graph to process

    Returns:
    - graph: networkx.Graph, the graph with isolated nodes connected
    """
    if len(graph.nodes) == 0 or len(graph.edges) == 0: return graph
    positions = np.array([graph.nodes[node]['point'] for node in graph.nodes])
    kdtree = cKDTree(positions)
    for node in graph.nodes:
        if graph.degree(node) == 0:
            _, nearest_neighbor = kdtree.query(positions[node], k=2)
            nearest_neighbor_index = nearest_neighbor[1] if nearest_neighbor[0] == node else nearest_neighbor[0]
            nearest_neighbor_id = list(graph.nodes)[nearest_neighbor_index]
            graph.add_edge(node, nearest_neighbor_id)
    return graph

def remove_small_components(graph, min_size=5):
    """
    Remove small components from the graph.

    Parameters:
    - graph: networkx.Graph, the graph from which small components will be removed
    - min_size: int, the minimum size of components to be retained in the graph

    Returns:
    - cleaned_graph: networkx.Graph, the graph with small components removed
    """
    cleaned_graph = graph.copy()
    kept_nodes = set()
    for component in list(nx.connected_components(graph)):
        if len(component) < min_size:
            for node in component:
                cleaned_graph.remove_node(node)
        else:
            kept_nodes.update(component)
    return cleaned_graph, kept_nodes

def fast_connect_components(graph):
    """
    Quickly connect all connected components in the graph using the improved KDTree method. 
    Compared with connect_components, this method connects each component to its closest component 
    in each round, greatly reducing the number of connected rounds and improving efficiency, and 
    is suitable for large figures or large components.
    Improvement strategy: 
        - Calculate its centroid for each component. 
        - Use KDTree to quickly find the most recent other components of each component. 
        - For each pair of components, look for the closest pair of nodes between them and add edges to connect them.
    Parameters: graph (networkx.Graph): The input graph, each node in the graph must contain the 'point' attribute, representing the spatial coordinates.
    Returns: networkx.Graph: The graph after all components have been connected through the nearest point.
    Notes:
        - If the picture itself is connected, no modification will be made. 
        - If the number of nodes in the graph is small or only one component, the performance benefits are not obvious. 
        - It is not guaranteed to constitute a global shortest connection, only a fast heuristic connection scheme.
    """
    components = list(nx.connected_components(graph))
    if len(components) <= 1:
        return graph
    # Calculate the center coordinates for each component
    centroids = []
    print(f'>> Total Components to Refine: {len(components)}')
    for comp in components:
        positions = np.array([graph.nodes[n]['point'] for n in comp])
        centroid = positions.mean(axis=0)
        centroids.append(centroid)
    # Build KDTree to search for recent components
    tree = cKDTree(centroids)
    _, nn_indices = tree.query(centroids, k=2)  # Nearest neighbor (skip yourself)
    connected = set()
    for i, j in enumerate(nn_indices[:, 1]):
        if (i, j) in connected or (j, i) in connected:
            continue
        connected.add((i, j))
        comp_i = list(components[i])
        comp_j = list(components[j])
        # Find the closest point pair between comp_i and comp_j
        points_i = np.array([graph.nodes[u]['point'] for u in comp_i])
        points_j = np.array([graph.nodes[v]['point'] for v in comp_j])
        tree_i = cKDTree(points_i)
        dists, indices = tree_i.query(points_j, k=1)
        idx_j = np.argmin(dists)
        idx_i = indices[idx_j]
        u = comp_i[idx_i]
        v = comp_j[idx_j]
        graph.add_edge(u, v)
    return graph

def connect_components(graph):
    """
    Quickly connect individual components in the graph using KD-Tree. The optimized accelerated version of connect_components.
    """
    while nx.number_connected_components(graph) > 1:
        components = list(nx.connected_components(graph))
        component_positions = []
        for component in components:
            positions = np.array([graph.nodes[node]['point'] for node in component])
            centroid = np.mean(positions, axis=0)
            component_positions.append(centroid)
        tree = cKDTree(component_positions)
        for i, pos in enumerate(component_positions):
            _, j = tree.query(pos, k=2)
            closest_pair = (i, j[1])
            break
        # Get the node with the most recent component
        component1, component2 = components[closest_pair[0]], components[closest_pair[1]]
        positions1 = np.array([graph.nodes[node]['point'] for node in component1])
        positions2 = np.array([graph.nodes[node]['point'] for node in component2])
        tree1 = cKDTree(positions1)
        dists, indices = tree1.query(positions2, k=1)
        # Determine the index of the minimum distance
        idx2 = np.argmin(dists)  # Index of position 2 of the minimum distance
        idx1 = indices[idx2]  # Take out the corresponding index from the index array at position 1
        # Select a node from the original component list
        node1 = list(component1)[idx1]
        node2 = list(component2)[idx2]
        closest_nodes = (node1, node2)
        # Add an edge connecting to the nearest node
        graph.add_edge(*closest_nodes)
    return graph

def connect_components_mst(graph):
    """
    Use the minimum spanning tree (MST) method to connect individual components in the graph. 
    The optimized accelerated version of connect_components. The composition effect is the same, but the acceleration is very obvious.
    """
    # Get all components
    components = list(nx.connected_components(graph))
    component_nodes = [list(comp) for comp in components]
    num_components = len(components)
    if num_components == 1:
        return graph  # If it is already connected, return directly
    # Construct a complete graph to represent the shortest distance between components
    distances = np.zeros((num_components, num_components))
    distances.fill(np.inf)
    # Calculate the centroid for each component
    centroids = []
    for nodes in component_nodes:
        positions = np.array([graph.nodes[node]['point'] for node in nodes])
        centroid = np.mean(positions, axis=0)
        centroids.append(centroid)
    # Optimize distance calculation using KD-Tree
    tree = cKDTree(centroids)
    for i in range(num_components):
        for j in range(i + 1, num_components):
            if distances[i, j] == np.inf:  # Avoid repeated calculations
                pos1 = np.array([graph.nodes[node]['point'] for node in component_nodes[i]])
                pos2 = np.array([graph.nodes[node]['point'] for node in component_nodes[j]])
                tree1 = cKDTree(pos1)
                dists, idxs = tree1.query(pos2, k=1)
                idx1 = idxs[np.argmin(dists)]  # Take the idx1 corresponding to the minimum value for multiple distances
                idx2 = np.argmin(dists)  # Take the index idx2 of the minimum value for multiple distances
                min_distance = dists.min()
                distances[i, j] = distances[j, i] = min_distance
                # Update the shortest distance node pair index
                closest_nodes = (component_nodes[i][idx1], component_nodes[j][idx2])
                distances[i, j] = distances[j, i] = dists.min()
    # Construct a complete graph of components
    complete_graph = nx.complete_graph(num_components)
    for i in range(num_components):
        for j in range(i + 1, num_components):
            complete_graph.add_edge(i, j, weight=distances[i, j])
    # Compute the minimum spanning tree
    mst = nx.minimum_spanning_tree(complete_graph, weight='weight')
    # Add edges to the original image to connect all components
    for edge in mst.edges():
        i, j = edge
        min_distance = np.inf
        best_pair = None
        # Find the closest node pair between two components
        for node1 in component_nodes[i]:
            for node2 in component_nodes[j]:
                pos1 = np.array(graph.nodes[node1]['point'])
                pos2 = np.array(graph.nodes[node2]['point'])
                distance = np.linalg.norm(pos1 - pos2)
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (node1, node2)
        graph.add_edge(*best_pair)
    return graph

def optimize_nodes_components(graph, min_size=20, image=None, show=False):
    batch_g = []
    kept_node_indices = []  # Record the node index retained by each graph
    for g in graph:
        g = connect_isolated_nodes(g)
        if show:
            print("2. Connect Isolated Nodes - Average degree (two-way):", sum(dict(g.degree()).values()) / g.number_of_nodes())
            visualize_graph_with_coordinates(g, title="2. Connect Isolated Nodes", image=image)
        g, kept = remove_small_components(g, min_size)
        if show:
            print("3. Remove Small Components - Average degree (two-way):", sum(dict(g.degree()).values()) / g.number_of_nodes())
            visualize_graph_with_coordinates(g, title="3. Remove Small Components", image=image)
        g = fast_connect_components(g)
        if show:
            print("4. Final Graph - Average degree (two-way):", sum(dict(g.degree()).values()) / g.number_of_nodes())
            visualize_graph_with_coordinates(g, title="4. Final Graph", image=image)
        batch_g.append(g)
        kept_node_indices.append(sorted(list(kept)))  # Convert to list Save order
    return batch_g, kept_node_indices


# External API—Dynamic Threshold
def build_optimize_graph_with_cosine_similarity(keypoints, descriptors, scores, radius=20, percentile=50, min_size=10, device=torch.device('cpu'), image=None, show=False):
    '''
    When percentage=0, min_size=1, it is the same as a fixed threshold. 
    Args:
        keypoints: Keypoint coordinate descriptors: Feature descriptors for keypoints 
        scores: Scores of keypoints radius: Points within the radius can generate edges 
        percentage: Feature descriptors have cosine similarity greater than the edges 
        min_size: The orphan subgraphs are smaller than the number of nodes that need to be deleted 
        device: GPU or CPU calculation 
    Returns: 
        Processed subgraphs
    '''
    graph = fast_build_graph_with_cosine_similarity(keypoints, descriptors, scores, radius, percentile)
    if show:
        print("1. Coarse Graph - Average degree (two-way):", sum(dict(graph[0].degree()).values()) / graph[0].number_of_nodes())
        visualize_graph_with_coordinates(graph[0], title="1. Coarse Graph", image=image)
    graph, kept_node_indices = optimize_nodes_components(graph, min_size=min_size, image=image, show=show)
    batch_graph = []
    for g in graph:
        points = torch.tensor(np.vstack([g.nodes[i]['point'] for i in g.nodes]), dtype=torch.float32, device=device)
        feats = torch.tensor(np.vstack([g.nodes[i]['feat'] for i in g.nodes]), dtype=torch.float32, device=device)
        scores = torch.tensor([g.nodes[i]['score'] for i in g.nodes], dtype=torch.float32, device=device)
        dgl_graph = dgl.from_networkx(g, device=device)
        dgl_graph.ndata['point'] = points
        dgl_graph.ndata['feat'] = feats
        dgl_graph.ndata['score'] = scores
        batch_graph.append(dgl_graph)
    return batch_graph, kept_node_indices

# External API—Fixed Threshold
def build_graph_from_keypoints(keypoints, descriptors, scores, radius, device=torch.device('cpu')):
    '''Fix the threshold, change the vertices within the threshold to generate edges'''
    graphs = create_graph_from_keypoints(keypoints, descriptors, scores, radius, device=torch.device('cpu'))
    return create_dgl_graph_from_batched_graph(graphs, device)


def build_graph_Delaunay(keypoints, descriptors, scores):
    keypoints = keypoints.cpu().numpy()
    descriptors = descriptors.permute(0, 2, 1).cpu().numpy()  # => (batch, number, dim)
    scores = scores.cpu().numpy()
    batch_graph = []
    for batched_idx in range(keypoints.shape[0]):
        kpts = keypoints[batched_idx]
        descs = descriptors[batched_idx]
        scrs = scores[batched_idx]
        graph = nx.Graph()
        for i in range(len(kpts)): graph.add_node(i, point=kpts[i], feat=descs[i], score=scrs[i])
        points = np.array([kp for kp in kpts])
        tri = Delaunay(points)
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    if not graph.has_edge(simplex[i], simplex[j]):
                        graph.add_edge(simplex[i], simplex[j])
        batch_graph.append(graph)
    return batch_graph

def build_graph_from_keypoints_Delaunay(keypoints, descriptors, scores, device=torch.device('cpu')):
    '''Fix the threshold, change the vertices within the threshold to generate edges'''
    graph = build_graph_Delaunay(keypoints, descriptors, scores)
    batch_graph = []
    for g in graph:
        points = torch.tensor(np.vstack([g.nodes[i]['point'] for i in g.nodes]), dtype=torch.float32, device=device)
        feats = torch.tensor(np.vstack([g.nodes[i]['feat'] for i in g.nodes]), dtype=torch.float32, device=device)
        scores = torch.tensor([g.nodes[i]['score'] for i in g.nodes], dtype=torch.float32, device=device)
        dgl_graph = dgl.from_networkx(g, device=device)
        dgl_graph.ndata['point'] = points
        dgl_graph.ndata['feat'] = feats
        dgl_graph.ndata['score'] = scores
        batch_graph.append(dgl_graph)
    return batch_graph








