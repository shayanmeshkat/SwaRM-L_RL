import random, math, os
import numpy as np
import networkx as nx
import cvxpy as cp


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
    
    print('w.value is:', W.value)
    print('q.value is:', q.value)
    print('A_full is:', A_full)
    # exit()
    return W.value, q.value, A_full

# def create_digraph_nodes_edge(N):

#     # Create a list of unique values (nodes)
#     nodes = list(range(1, N + 1))

#     # Shuffle the nodes randomly
#     random.shuffle(nodes)

#     # Repeat the shuffled nodes to create vectors s and t
#     s = nodes + nodes
#     t = random.sample(nodes, len(nodes)) + random.sample(nodes, len(nodes))
    
#     return s, t

# def create_digraph_nodes_edge(N):
#     # Create a list of unique values (nodes)
#     nodes = list(range(1, N + 1))
    
#     # Initialize empty lists for source and target nodes
#     s = []
#     t = []
    
#     # First, create a cycle to ensure connectivity
#     for i in range(N):
#         s.append(nodes[i])
#         t.append(nodes[(i + 1) % N])
    
#     # Then add additional random edges to reach target_num edges per node
#     target_num = 2
#     additional_edges = target_num - 1  # We already have 1 edge per node from the cycle
    
#     for node in nodes:
#         # Get possible targets (excluding self and existing connections)
#         existing_targets = [t[i] for i, src in enumerate(s) if src == node]
#         possible_targets = [n for n in nodes if n != node and n not in existing_targets]
        
#         if len(possible_targets) >= additional_edges:
#             targets = np.random.choice(possible_targets, additional_edges, replace=False)
#             s.extend([node] * additional_edges)
#             t.extend(targets)
    
#     print(s, t)
#     return s, t

# def select_random_neighbor(adjacency_matrix):
#     num_agents = adjacency_matrix.shape[0]
    
#     print('adjacency matrix:', adjacency_matrix)
#     # Choose a random agent
#     random_agent = np.random.randint(num_agents)

#     print('adjacency matrix of random agent', adjacency_matrix[random_agent])
#     print('non zero elements:', np.nonzero(adjacency_matrix[random_agent]))
#     print('non zero elements index 1:', np.nonzero(adjacency_matrix[random_agent])[0])

#     # exit()
#     # Find the neighbors of the random agent
#     neighbors = np.nonzero(adjacency_matrix[random_agent])[0]

#     if len(neighbors) == 0:
#         # If the random agent has no neighbors, return None
#         return None, None

#     # Choose a random neighbor from the list of neighbors
#     random_neighbor = np.random.choice(neighbors)
#     print('random neighbor:', random_neighbor)
#     exit()
#     return random_agent, random_neighbor

def select_random_neighbor(A):
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

# def create_digraph_nodes_edge(N):

#     # Create a list of unique values (nodes)
#     nodes = list(range(1, N + 1))

#     # Shuffle the nodes randomly
#     random.shuffle(nodes)

#     # Repeat the shuffled nodes to create vectors s and t
#     s = nodes + nodes
#     t = random.sample(nodes, len(nodes)) + random.sample(nodes, len(nodes))
#     # print(s, t)
#     # exit()
#     return s, t

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

class ConstantRewardFunction:
    def __init__(self, reward):
        self.reward = reward
    
    def get_reward(self):
        return self.reward

class RewardMachine_swarm:
    def __init__(self):
        self.transitions = {}
    
    def add_transition(self, from_state, to_state, condition, reward_function):
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        self.transitions[from_state].append((to_state, condition, reward_function))
    
    def get_reward(self, current_state, action):
        if current_state in self.transitions:
            for transition in self.transitions[current_state]:
                to_state, condition, reward_function = transition
                if self.check_condition(condition, action):
                    return reward_function.get_reward(), to_state
        return 0, current_state
    
    def check_condition(self, condition, action):
        if '&' in condition:
            parts = condition.split('&')
            for part in parts:
                if part.startswith('!'):
                    if part[1:] in action:
                        return False
                else:
                    if part not in action:
                        return False
            return True
        else:
            return condition in action

def read_reward_machine(filename):
    reward_machine = RewardMachine_swarm()
    print(os.getcwd())
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('('):
            parts = line.split(',')
            from_state = int(parts[0][1:])
            to_state = int(parts[1])
            condition = parts[2].strip("'")
            reward_value = int(parts[3].strip("ConstantRewardFunction()"))
            reward_function = ConstantRewardFunction(reward_value)
            reward_machine.add_transition(from_state, to_state, condition, reward_function)

    return reward_machine

# Read the reward machine from the text file
# reward_machine = read_reward_machine('../experiments/office/reward_machines/t1.txt')



def labeling_function(agent_x, agent_y, current_state, target_list, error_upp_bound, label, reward_machine):
 

    # Define a dictionary to map X and Y positions to action labels
    position_to_action = {
        (target_list[0][0], target_list[0][1]): 'a',
        (target_list[1][0], target_list[1][1]): 'b',
#         (1, 2): 'b',
#         (2, 2): 'True',
#         (3, 3): 'a',
    }
#     print(agent_x - target_list[0][0])
    
    if label == 'a' and np.abs(agent_x - target_list[0][0]) <= error_upp_bound and np.abs(agent_y - target_list[0][1]) <= error_upp_bound:
#     if label == 'a' and np.abs(agent_x - target_list[0][0]) <= error_upp_bound:
        
        
#         print('the label is', label)
        reward, next_state = reward_machine.get_reward(current_state, label)
#         reward = 0.02
#         print('reward is', reward, 'next state is', next_state)
        label = position_to_action[target_list[1]]
#     print
        
        
        
    elif label == 'b' and np.abs(agent_x - target_list[1][0]) <= error_upp_bound and np.abs(agent_y - target_list[1][1]) <= error_upp_bound:
#     elif label == 'b' and np.abs(agent_x - target_list[1][0]) <= error_upp_bound:
#         print('target list of 2nd task is', target_list[1])
#         print('its label is', position_to_action[target_list[1]])
        
        print('2nd task label', label)
        reward, next_state = reward_machine.get_reward(current_state, label)
#         print(reward)
        label = 'True'
    else:
        if label == '!a&!b':
            reward = 0
            next_state = 1
            label = 'a'
            
        elif current_state == 1 and label == 'b':
            label = 'b'
            reward=0
            next_state = 1
            
        elif current_state == 2 and label == 'True':
            label = 'True'
            reward=0
            next_state = 2
        else: 
            label=='a'
            reward = 0
            next_state = 1
            label = 'a'


    return reward, next_state, label


def state_transfer_2_dim(s, grid_width, grid_height, agents_num):
    s_modified = [[0,0] for _ in range(agents_num)]
    for i in range(agents_num):
        s_modified[i][1] = s[i]//grid_width
        s_modified[i][0] = np.mod(s[i], grid_height)
    return s_modified


def create_list_of_dicts(N):
    # Initialize an empty list to store dictionaries
    # q_list = [{} for i in range(N)]
    q_list = [np.zeros([8, 8, 4, 4])for i in range(N)]
    
    return q_list

def create_tuple_with_selected_index(length, selected_index):
    if selected_index < 0 or selected_index >= length:
        raise ValueError("selected_index is out of bounds")

    # Create a tuple with all zeros
    result = tuple(0 for _ in range(length))
    
    # Set the selected index to 1
    result = result[:selected_index] + (1,) + result[selected_index + 1:]
    
    return result

def flatten_nested_tuple(nested_tuple):
    flattened = []
    for item in nested_tuple:
        if isinstance(item, tuple):
            flattened.extend(flatten_nested_tuple(item))
        else:
            flattened.append(item)
    return tuple(flattened)

# if __name__ == '__main__':
    # print('hi')
    # q_tet = create_list_of_dicts(4, 2,2,3, 4)
    # print(q_tet[0](0,0,1,0,0))


def calculate_averages(current_state_list):
    if not current_state_list:
        return None, None
    
    # Convert the list of tuples to a NumPy array for easier calculations
    data_array = np.array(current_state_list)
    
    # Calculate the x and y averages separately
    x_average = np.mean(data_array[:, 0])
    y_average = np.mean(data_array[:, 1])
    
    return x_average, y_average

def calculate_variances(current_state_list):
    if not current_state_list:
        return None, None

    # Convert the list of tuples to a NumPy array for easier calculations
    data_array = np.array(current_state_list)

    # Calculate the variances for x and y values separately
    x_variance = np.var(data_array[:, 0], ddof=0)  # Set ddof to 0 for population variance
    y_variance = np.var(data_array[:, 1], ddof=0)

    return x_variance, y_variance

def get_one_hot_vector(events):
    # NOTE: Get the list of labels from the environment
    all_labels = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5}
    
    one_hot_vector = [[0 for i in range(len(all_labels))] for i in range(len(events))]
    for i in range(len(events)):
        if all_labels.get(events[i], None)!= None:
            one_hot_vector[i][all_labels[events[i]]] = 1
    # print(one_hot_vector)
    return one_hot_vector

