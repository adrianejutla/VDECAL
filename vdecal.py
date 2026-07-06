import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def binarySplit(X, indices, random_state=42):
    """
    Splits a set of points into two clusters using KMeans
    
    Parameters:
    X: numpy matrix with all points
    indices: array of indices referencing points in X
    random_state: seed for reproducibility
    
    Returns:
    indices1, indices2: arrays of indices divided into two clusters
    """
    # Extract the points corresponding to the indices
    points = X[indices]
    
    # Apply KMeans with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=random_state, init="k-means++", n_init=10, max_iter=10)
    labels = kmeans.fit_predict(points)
    
    # Split the original indices according to clusters
    indices1 = indices[labels == 0]
    indices2 = indices[labels == 1]
    
    return indices1, indices2


def partitionStageRandom(X, ncores, random_state=42):
    """
    Random partitions the dataset 
    
    Parameters:
    X: numpy matrix with all points
    ncores: number of partition
    random_state: seed for reproducibility
    
    Returns:
    partitions: list of arrays with indices of each final partition
    partition_num: label for the last partition 
    """
    
    partition_labels = np.random.randint(0, ncores, size=len(X))
    
    return partition_labels, ncores-1



def partitionStageKMeans(X, t, random_state=42):
    """
    Recursively partitions the dataset using binary splits until 
    all clusters have size <= max_size
    
    Parameters:
    X: numpy matrix with all points
    t: maximum cluster size
    random_state: seed for reproducibility
    
    Returns:
    partitions: list of arrays with indices of each final partition
    partition_num: label for the last partition 
    """
    # Initialize with all indices
    all_indices = np.arange(len(X))
    partitions = []
    
    # Use LIFO stack for processing
    pending = [all_indices]
    
    while pending:
        current_indices = pending.pop()
        
        if len(current_indices) > t:
            # Split into two clusters
            indices1, indices2 = binarySplit(X, current_indices, random_state)
            
            # Add to stack for further processing
            pending.append(indices1)
            pending.append(indices2)
        else:
            # Cluster is small enough, add to results
            partitions.append(current_indices)
    
    partition_labels = np.zeros(len(X), dtype=int)
    partition_num = 0
    for partition_num, indices in enumerate(partitions):
        partition_labels[indices] = partition_num
    
    return partition_labels, partition_num




def calcular_MdenVar(Madj, ROS_D):
    """
    Calcula la matriz MdenVar según la fórmula especificada.
    
    Parámetros:
    Madj: matriz de adyacencia (boolean) de tamaño n x n
    ROS_D: vector de valores float de tamaño n
    
    Retorna:
    MdenVar: matriz de tamaño n x n con los valores calculados
    """
    n = len(ROS_D)
    MdenVar = np.full((n, n), 3.0)  # Inicializar con NaN
    
    for j in range(n):
        # Encontrar los índices donde j tiene True en la matriz de adyacencia
        indices_true = np.where(Madj[:, j])[0]
        
        if len(indices_true) > 0:
            # Obtener los valores de ROS_D correspondientes a esos índices
            valores_j = ROS_D[indices_true]
            
            # Calcular media y desviación estándar
            media_j = np.mean(valores_j)
            std_j = np.std(valores_j)
            
            # Si la desviación estándar es 0, evitar división por cero
            if std_j == 0:
                std_j = 1e-10  # Valor muy pequeño para evitar división por cero
            
            # Calcular MdenVar solo para las filas i donde Madj[i,j] es True
            for i in indices_true:
                MdenVar[i, j] = np.abs((ROS_D[i] - media_j) / std_j)
                
    return MdenVar




def generateRelevantObjectSubsets(X, parts, p, eps, minPts=1):
    """
    Parameters:
    X: numpy matrix with all points
    parts: array with partition labels for each point
    p: partition number to analyze
    eps: radius for neighborhood search
    minPts: minimum points required to form a neighborhood
    
    Returns:
    Neighborhoods: list of arrays with indices of points in each neighborhood
    Centroids: list of centroid points for each neighborhood
    Radii: list of maximum radii for each neighborhood
    Noise: list of global indices marked as noise (not in any neighborhood)
    """
    # Get points and their global indices for partition p
    idpts = np.where(parts == p)[0]
    pts = X[idpts]
    
    if len(pts) == 0:
        return [], [], [], []
    
    # Create KDTree for efficient neighborhood search
    tree = cKDTree(pts)
    
    # Track processed points and noise
    procesados = set()
    Neighborhoods = []
    Centroids = []
    Radii = []
    Noise = set(idpts)  # Start with all points as potential noise
    Densities = []
    
    
    idx = 0
    while len(procesados) < len(pts):
        # Find next unprocessed point
        while idx in procesados and idx < len(pts):
            idx += 1
        if idx >= len(pts):
            break
            
        centroid = pts[idx]
        neighborhood_indices = tree.query_ball_point(centroid, eps)
        
        # Only create neighborhood if it has enough points
        if len(neighborhood_indices) >= minPts:
            # Mark these points as processed
            procesados.update(neighborhood_indices)
            
            # Calculate neighborhood properties
            neighborhood_points = pts[neighborhood_indices]
            distances = np.linalg.norm(neighborhood_points - centroid, axis=1)
            max_radius = np.max(distances)
            density = len(neighborhood_points)
            if max_radius > 0:
                density    = len(neighborhood_points) / max_radius
                    
            
                
            # Get global indices for this neighborhood
            global_indices = idpts[neighborhood_indices]
            
            # Store results
            Neighborhoods.append(global_indices)
            Centroids.append(centroid)
            Radii.append(max_radius)
            Densities.append(density)
            
            
            # Remove these points from noise (they belong to a neighborhood)
            Noise -= set(global_indices)
        else:
            # Mark single point as processed but don't create neighborhood
            procesados.add(idx)
            # This point remains in Noise set
        
        idx += 1
    
    return Neighborhoods, Centroids, Radii, Densities, list(Noise)



def generaClusters (C, R, D, delta):
    MDis = cdist(C, C)
    Madj = MDis <= np.add.outer(R, R)
    n = len(R)
    #construccion del grafo
    G = nx.Graph()
    for i in range(n):
        G.add_node(i  ) 


    for j in range(n):
        indices_true = np.where(Madj[:, j])[0]
        if len(indices_true) > 0:
            valores_j = D[indices_true]
            media_j = np.mean(valores_j)
            std_j = np.std(valores_j)
            if std_j == 0:
                std_j = 1e-10  # Valor muy pequeño para evitar división por cero
            for i in indices_true:
                zscore = np.abs((D[i] - media_j) / std_j) 
                if G.has_edge(i,j):
                    G.edges[(i,j)]['weight'] = np.min([ G.edges[(i,j)]['weight'] , zscore])
                else:
                    G.add_edge(i, j, weight = zscore)
                 
        
    mst = nx.minimum_spanning_tree(G)
    filtered_mst = mst.copy()
    edges_to_remove = []
    for u, v, data in mst.edges(data=True):
        if not data['weight'] < delta: 
            edges_to_remove.append((u, v))
            
    filtered_mst.remove_edges_from(edges_to_remove)
    components = list(nx.connected_components(filtered_mst))

    return components, filtered_mst



def assign_labels(X, C, labels_C, eps):
    """
    Assign labels to points in X based on minimum distance to points in C.
    
    Parameters:
    X: array of shape (n_points_X, n_dimensions) - Points to classify
    C: array of shape (n_points_C, n_dimensions) - Reference points
    labels_C: array of shape (n_points_C,) - Labels of reference points
    eps: float - Maximum distance threshold for label assignment
    
    Returns:
    labels_X: array of shape (n_points_X,) - Assigned labels
    """
    
    # Calculate distances between each point in X and each point in C
    # Using broadcasting to efficiently compute all distances
    differences = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=2))
    
    # Find minimum distance and index of closest point in C
    min_distances = np.min(distances, axis=1)
    min_indices = np.argmin(distances, axis=1)
    
    # Assign labels based on eps threshold
    labels_X = np.full(X.shape[0], -1, dtype=labels_C.dtype)
    
    # For points within threshold, assign corresponding label
    within_threshold_mask = min_distances <= eps
    labels_X[within_threshold_mask] = labels_C[min_indices[within_threshold_mask]]
    
    return labels_X

    
def VDECAL(X, t, eps, minPts, delta, partition_type="Kmeans++", random_state=42):
    parts, last = None, None
    if partition_type=="Random":
        parts , last = partitionStageRandom(X, t, random_state=random_state)
    else:
        parts , last = partitionStageKMeans(X, t, random_state=random_state)
    
#     print ("Iniciando calculo de particiones")
    ROS_C, ROS_R, ROS_D = [], [], []
    for p in range(last+1):
        N, C, R, D, Noise = generateRelevantObjectSubsets(X, parts, p, eps, minPts)
        
        ROS_C += C
        ROS_R += R
        ROS_D += D
    
    ROS_C = np.array(ROS_C)
    ROS_D = np.array(ROS_D)
    ROS_R = np.array(ROS_R)
    
#     #print ("Iniciando calculo de ROS")
    components, filtered_mst = generaClusters(ROS_C,ROS_R,ROS_D, delta)
    labels_C = np.zeros(len(ROS_C))
    for id_cl, cl in enumerate(components):
        for c in cl:
            labels_C[c] = id_cl
    
    
    return assign_labels(X, ROS_C, labels_C, eps)

    
    
