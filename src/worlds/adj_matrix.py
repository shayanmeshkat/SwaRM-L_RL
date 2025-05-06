import numpy as np
import networkx as nx
import cvxpy as cp
import matplotlib.pyplot as plt


def create_digraph_nodes_edge(N):
    # Create a list of unique values (nodes)
    nodes = list(range(1, N + 1))
    
    # Initialize empty lists for source and target nodes
    s = []
    t = []
    
    # First, create a cycle to ensure connectivity
    for i in range(N):
        s.append(nodes[i])
        t.append(nodes[(i + 1) % N])
    
    # Then add additional random edges to reach target_num edges per node
    target_num = 2
    additional_edges = target_num - 1  # We already have 1 edge per node from the cycle
    
    for node in nodes:
        # Get possible targets (excluding self and existing connections)
        existing_targets = [t[i] for i, src in enumerate(s) if src == node]
        possible_targets = [n for n in nodes if n != node and n not in existing_targets]
        
        if len(possible_targets) >= additional_edges:
            targets = np.random.choice(possible_targets, additional_edges, replace=False)
            s.extend([node] * additional_edges)
            t.extend(targets)
    
    print(s, t)
    return s, t


def get_topology_matrix(N ,s, t):
    # Generate random edges for the graph
#     edges = random.sample(range(1, N+1), N)
#     s = edges
#     t = edges[1:] + [edges[0]]  # Ensure the last node points to the first node
    
    
    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(zip(s, t))

    # Forming the adjacency matrix
    A = nx.adjacency_matrix(G).todense()

    A_full = A.T | A

    # Initialize the matrix W_array for each agent
    W_array = np.zeros((N, N, N, N))

    for i in range(N):
        for j in range(N):
            e_i = np.zeros(N)
            e_j = np.zeros(N)
            e_i[i] = 1
            e_j[j] = 1
            W_array[:, :, i, j] = np.eye(N) - np.outer(e_i - e_j, e_i - e_j) / 2

    q = cp.Variable()
    # Initialize W_neighbor with the correct dimensions
    W_neighbor = cp.Variable((N, N))
    W = cp.Variable((N, N), symmetric=True)

    constraints = [W_neighbor >= 0]

    constraints += [cp.sum(1 - cp.multiply(A_full, W_neighbor)) == 0]  # Use cp.multiply for element-wise multiplication

    weighted_sum = 0
    for i in range(N):
        for j in range(N):
            weighted_sum += W_neighbor[i, j] * W_array[:, :, i, j]

    # Define the semidefinite constraint using cp.Variable
    SDP_var = cp.Variable((N, N), symmetric=True)
    constraints += [SDP_var >> np.eye(N)]  # Here, we specify that SDP_var is positive semidefinite

    constraints += [W == SDP_var - 1/N * np.eye(N) - q * np.eye(N)]
    constraints += [cp.sum(W, axis=1) == 1]

    objective = cp.Minimize(q)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    # print('w.value is:', W.value)
    # print('q.value is:', q.value)
    print('A_full is:', A_full)
    # exit()
    return W.value, q.value, A_full

def visualize_digraph(s, t, N):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_nodes_from(range(1, N+1))
    
    # Add edges from source and target lists
    edges = list(zip(s, t))
    G.add_edges_from(edges)
    
    # Set up the plot
    plt.figure(figsize=(7, 7))
    
    # Create a circular layout
    pos = nx.circular_layout(G)
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, 
            node_color='lightblue',
            node_size=500,
            arrowsize=20,
            font_size=16,
            font_weight='bold')
    
    plt.title("Directed Graph Visualization")
    plt.show()

def select_random_agent_and_neighbor(A):
    """
    Select a random agent and one of its neighbors based on adjacency matrix.
    
    Args:
        A: NxN adjacency matrix where A[i,j]=1 if agent i follows j
        
    Returns:
        tuple: (agent_id, neighbor_id), or (None, None) if no valid selection possible
    """
    N = A.shape[0]
    agent_id = np.random.randint(0, N)
    
    # Find neighbors (nodes that the agent follows)
    neighbors = np.where(A[agent_id] == 1)[0]
    
    if len(neighbors) == 0:
        return None, None
        
    # Randomly select one neighbor
    neighbor_id = np.random.choice(neighbors)
    
    return agent_id, neighbor_id


N = 10
s, t = create_digraph_nodes_edge(N)
W, q, A = get_topology_matrix(N, s, t)
agent, neighbor = select_random_agent_and_neighbor(A)
print('agent:', agent+1, 'neighbor:', neighbor+1)

visualize_digraph(s, t, N)