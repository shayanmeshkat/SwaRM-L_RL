if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')

from worlds.game_objects import Actions
import random, math, os
import numpy as np
import networkx as nx
import cvxpy as cp

# from tester.tester import Tester
# from tester.livetester import LiveTester

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class OfficeWorldParams:
    def __init__(self):
        pass

class OfficeWorld:

    def __init__(self, params, num_agents):
        self.env_game_over = False
        self.params = params
        self.num_agents = num_agents
        self._load_map()

    def execute_action(self, action_list):
        """
        We execute 'action' in the game
        """

        for agent_ind in range(self.num_agents):
            self.agent[agent_ind] = self.xy_MDP_slip(agent_ind,action_list[agent_ind],1) # progresses in x-y system

    def xy_MDP_slip(self,agent_ind, agent_action, p):
        x,y = self.agent[agent_ind]
        slip_p = [p,(1-p)/2,(1-p)/2]
        check = random.random()

        # up    = 0
        # right = 1 
        # down  = 2 
        # left  = 3 

        if (check<=slip_p[0]):
            action_ = agent_action

        elif (check>slip_p[0]) & (check<=(slip_p[0]+slip_p[1])):
            if agent_action == 0: 
                action_ = 3
            elif agent_action == 2: 
                action_ = 1
            elif agent_action == 3: 
                action_ = 2
            elif agent_action == 1: 
                action_ = 0

        else:
            if agent_action == 0: 
                action_ = 1
            elif agent_action == 2: 
                action_ = 3
            elif agent_action == 3: 
                action_ = 0
            elif agent_action == 1: 
                action_ = 2

        action_string = Actions(action_)
        if (x,y,action_string) not in self.forbidden_transitions:
            if action_string == Actions.up:
                y+=1
            if action_string == Actions.down:
                y-=1
            if action_string == Actions.left:
                x-=1
            if action_string == Actions.right:
                x+=1

        # self.action_[agent_ind] = action_
        return (x,y)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.actions

    def get_last_action(self):
        """
        Returns agent's last action
        """
        return self.action_

    def get_true_propositions(self, gm_x_estimate, gm_y_estimate, error_upp_bound,
                              a_full, s_x, s_y):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ["" for i in range(self.num_agents)]
        
        gm_x_neighbors, gm_y_neighbors = [[] for i in range(self.num_agents)], [[] for i in range(self.num_agents)]

        for i in range(self.num_agents):
           ret[i] = self.get_label(gm_x_estimate[i], gm_y_estimate[i], error_upp_bound)
        
        ret_own = ["" for i in range(self.num_agents)]
        for i in range(self.num_agents):
            ret_own[i] = self.get_label(self.agent[i][0], self.agent[i][1], error_upp_bound)

        for i in range(self.num_agents):
            neighbors_list = np.where(a_full[i] == 1)[0]
            # print('neighbors_list before agent itself:', neighbors_list)

            neighbors_list = np.append(neighbors_list, i)
            # print('neighbors_list after agent itself:', neighbors_list)
            # exit()
            # x_neighbors = np.array([s_x[j] for j in neighbors_list])
            # mean_x = int(np.round(np.mean(x_neighbors)))

            # # Get y coordinates of neighbors and average them
            # y_neighbors = np.array([s_y[j] for j in neighbors_list])
            # mean_y = int(np.round(np.mean(y_neighbors)))
            
            # ret[i] = self.get_label(mean_x, mean_y, error_upp_bound)

            x_agents = np.array([self.agent[j][0] for j in neighbors_list])
            mean_x = np.mean(x_agents)

            # Get y coordinates of neighbors and average them
            y_agents = np.array([self.agent[j][1] for j in neighbors_list])
            mean_y = np.mean(y_agents)
            
            #ret[i] = self.get_label(mean_x, mean_y, error_upp_bound)

        # ret_true = ["" for i in range(self.num_agents)]
        
        # for i in range(self.num_agents):
        #     ret_true[i] = self.get_label(gm_x_estimate[i], gm_y_estimate[i], error_upp_bound)
        # print('agents positions are:', self.agent)
        gm_x_true, gm_y_true = self.calculate_averages(self.agent)
        # print('gm_x_true:', gm_x_true, 'gm_y_true:', gm_y_true)
        ret_true = self.get_label(gm_x_true, gm_y_true, error_upp_bound)
        # for i in range(self.num_agents):
        #     ret_true[i] = self.get_label(gm_x_true, gm_y_true, error_upp_bound)
        return ret, ret_true, ret_own
    
    def get_label(self, agent_gm_x, agent_gm_y, error_upper_bound):
        
        # Get the closest object location based on current state
        min_dist = 1000000
        label = ""
        for key, val in self.objects.items():
            # print(f"Key: {key}; Value: {val}")
            temp_dist = ((agent_gm_x - key[0])**2 + (agent_gm_y - key[1])**2)**0.5
            if temp_dist<min_dist:
                closest_point = key
                min_dist = temp_dist



        # Check if your estimate of current state of gm is correct
        if np.abs(agent_gm_x - closest_point[0]) <= error_upper_bound and np.abs(agent_gm_y - closest_point[1]) <= error_upper_bound:
            # Then give the label based on the estiamtion
            label = self.objects[closest_point]

        
        return label

    def calculate_averages(self, current_state_list):
        if not current_state_list:
            return None, None
        
        # Convert the list of tuples to a NumPy array for easier calculations
        data_array = np.array(current_state_list)
        
        # Calculate the x and y averages separately
        x_average = np.mean(data_array[:, 0])
        y_average = np.mean(data_array[:, 1])
        
        return x_average, y_average

    def get_state(self):
        return None # we are only using "simple reward machines" for the craft domain
    
    def get_state_vector(self):
        x,y = [0 for i in range(self.num_agents)],[0 for i in range(self.num_agents)]
        for i in range(self.num_agents):
            x[i],y[i] = self.agent[i]
        
        return (x,y) 

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        x,y = [0 for i in range(self.num_agents)],[0 for i in range(self.num_agents)]
        for i in range(self.num_agents):
            x[i],y[i] = self.agent[i]
        N,M = 8,8
        ret_all = []
        # ret_temp = np.zeros((N,M), dtype=np.float64)
        for i in range(self.num_agents):
            ret_temp = np.zeros((N,M), dtype=np.float64)
            ret_temp[x[i],y[i]] = 1
            ret_all.append(ret_temp.ravel())
        # print(ret_all)
        return ret_all # from 2D to 1D (use a.flatten() is you want to copy the array)

    # # The following methods create the map ----------------------------------------------


    def _load_map(self):
        # Creating the map
        self.objects = {}
        # self.objects[(2,2)] = "a"
        # self.objects[(3,4)] = "b"
        # self.objects[(5,6)] = "c"

        self.objects[(3,4)] = "a"
        self.objects[(5,6)] = "b"

        # Adding the agent
        self.agent = [(0,0) for i in range(self.num_agents)]
        self.actions = [Actions.up.value,Actions.right.value,Actions.down.value,Actions.left.value]

        
        # self.num_agents = self.testing_params

        

        # Adding walls
        # Adding walls
        self.forbidden_transitions = set()
        # for x in range(8):
        #     self.forbidden_transitions.add((x,0,Actions.down))
        #     self.forbidden_transitions.add((x,7,Actions.up))

        # for y in range(8):
        #     self.forbidden_transitions.add((0,y,Actions.left))
        #     self.forbidden_transitions.add((7,y,Actions.right))
        for x in range(8):
            for y in [0]:
                self.forbidden_transitions.add((x,y,Actions.down))
            for y in [7]:
                self.forbidden_transitions.add((x,y,Actions.up))

        for y in range(8):
            for x in [0]:
                self.forbidden_transitions.add((x,y,Actions.left))
            for x in [7]:
                self.forbidden_transitions.add((x,y,Actions.right))
def play():
    from reward_machines.reward_machine import RewardMachine

    # commands
    str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value}
    params = OfficeWorldParams()

    # play the game!
    tasks = ["../../experiments/office/reward_machines/t%d.txt"%i for i in [1,2,3,4]]
    reward_machines = []
    for t in tasks:
        reward_machines.append(RewardMachine(t))
    for i in range(len(tasks)):
        print("Running", tasks[i])

        game = OfficeWorld(params) # setting the environment
        rm = reward_machines[i]  # setting the reward machine
        s1 = game.get_state()
        u1 = rm.get_initial_state()
        while True:
            # Showing game
            game.show()
            print("Events:", game.get_true_propositions())
            #print(game.getLTLGoal())
            # Getting action
            print("u:", u1)
            print("\nAction? ", end="")
            a = input()
            print()
            # Executing action
            if a in str_to_action:
                game.execute_action(str_to_action[a])

                # Getting new state and truth valuation
                s2 = game.get_state()
                events = game.get_true_propositions()
                u2 = rm.get_next_state(u1, events)
                r = rm.get_reward(u1,u2,s1,a,s2)
                
                # Getting rewards and next states for each reward machine
                rewards, next_states = [],[]
                for j in range(len(reward_machines)):
                    j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, events)
                    rewards.append(j_rewards)
                    next_states.append(j_next_states)
                
                print("---------------------")
                print("Rewards:", rewards)
                print("Next States:", next_states)
                print("Reward:", r)
                print("---------------------")
                
                if game.env_game_over or rm.is_terminal_state(u2): # Game Over
                    break 
                
                s1 = s2
                u1 = u2
            else:
                print("Forbidden action")
        game.show()
        print("Events:", game.get_true_propositions())


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

    return W.value, q.value, A_full



# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    # play()
    N = 4
    s = [1, 1, 2, 3]
    t = [2, 4, 3, 4]
    adjacency_matrix, lambda_2, a_full = get_topology_matrix(N, s ,t)
    print(a_full)
