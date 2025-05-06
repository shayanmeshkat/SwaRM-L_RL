import numpy as np
import random, time
import itertools
from automata_learning_utils import al_utils
# import automata_learning_utils.al_utils
from worlds.game import *
from reward_machines.reward_machine import RewardMachine
from automata_learning.Traces import Traces
from tester.tester import Tester
from tester.livetester import LiveTester
from tester.timer import Timer
from worlds.adjacency_matrix import *
import shutil
import os
import subprocess
import csv
import time
import copy
import pandas as pd

def run_aqrm_task(epsilon, env, learned_rm_file, tester_true, tester_learned, curriculum, show_print, is_rm_learned, currentstep, previous_testing_reward, q,num_agents, a_full,
                  #swarm_reward_machine
                  ):
    """
    This code runs one training episode.
        - rm_file: It is the path towards the RM machine to solve on this episode
        - environment_rm: an environment reward machine, the "true" one, underlying the execution
    """
    # Initializing parameters and the game
    learning_params = tester_learned.learning_params
    testing_params = tester_learned.testing_params
    # print('agent number is', testing_params.agents_num)
    """
     here, tester holds all the machines. we would like to dynamically update the machines every so often.
     an option might be to read it every time a new machine is learnt
     """
    

    task_params = tester_learned.get_task_params(learned_rm_file) 
    task = Game(task_params, num_agents)
    actions = task.get_actions()
    num_actions = len(actions)
    num_steps = learning_params.max_timesteps_per_task
    training_reward = 0
    is_conflicting = 1 #by default add traces
    is_conflicting_list = [1 for i in range(testing_params.agents_num)]
    testing_reward = None #initialize

    # Getting the initial state of the environment and the reward machine
    rm_learned = tester_learned.get_hypothesis_machine()
    rm_true = tester_true.get_reward_machines()[0]

    # u1_swarm_estimate = [rm_learned.get_initial_state() for _ in range(testing_params.agents_num)] 
    u1_swarm_list = [rm_learned.get_initial_state() for _ in range(testing_params.agents_num)] 
    u1_swarm_list_for_learn = [rm_learned.get_initial_state() for _ in range(testing_params.agents_num)]
    # u1_swarm = rm_learned.get_initial_state() 

    u1_swarm_true = rm_true.get_initial_state()

    alpha = 0.001
    gamma = 0.9
    w = 0
    T = 100

    # Starting interaction with the environment
    if show_print: print("Executing", num_steps)
    all_events = []
    a=[0 for i in range(num_agents)]
    reward = [0 for i in range(testing_params.agents_num)]
    reward_true = [0 for i in range(testing_params.agents_num)]
    # print('the s is',s)
    update_q = [1 for i in range(testing_params.agents_num)]

    ############################################################################
    # NOTE:Change the hard code initial location and get that location from the environment
    gm_x_estimate = [0 for _ in range(testing_params.agents_num)]
    gm_y_estimate = [0 for _ in range(testing_params.agents_num)]
    error_upp_bound = 1.5

    ############################################################################

    s_x, s_y = task.get_state_vector()
    # print('s_x is:', s_x)
    # print('s_y is:', s_y)

    estimate_label, true_label, events_own = task.get_true_propositions(gm_x_estimate, gm_y_estimate, error_upp_bound, a_full, s_x, s_y) 
    label_one_hot_vector = get_one_hot_vector(estimate_label)
    # print('label one hot vector is', label_one_hot_vector)
    # exit()
    # print(f"Initial label vector: {label_one_hot_vector}")
    current_state_AS = []
    for i in range(num_agents):
        current_state_AS.append(tuple([s_x[i], s_y[i]] + label_one_hot_vector[i]))

    # print('current state AS before start is:', current_state_AS)
    # exit()
    for t in range(num_steps):
        currentstep += 1
        # s_x, s_y = task.get_state_vector()
        # s_old = state_transfer_2_dim(s, 8, 8, testing_params.agents_num)
        # print('q_value is:', q[i])
        # exit()

        eta_list = np.zeros([num_agents, 2])
        eta_list_new = np.zeros([num_agents, 2])


        for i in range(num_agents):
            if random.random() < 0.3:
                a[i] = random.choice(actions)
            else:
                # if max(q[i][s[i]])==0:
                # print('arg is', current_state_AS[i])
                # print('value is', q[i].get(current_state_AS[i],[0,0,0,0]))

                neighbors_list = np.where(a_full[i] == 1)[0]
                
                # Add the agent itself to the list
                neighbors_list = np.append(neighbors_list, i)

                # Get x coordinates of neighbors and average them
                x_neighbors = np.array([s_x[j] for j in neighbors_list])
                mean_x = int(np.round(np.mean(x_neighbors)))

                # Get y coordinates of neighbors and average them
                y_neighbors = np.array([s_y[j] for j in neighbors_list])
                mean_y = int(np.round(np.mean(y_neighbors)))

                # Store results
                eta_list[i] = [mean_x, mean_y]

                # if max(q[i][int(eta_list[i][0])][int(eta_list[i][1])][u1_swarm_list[i]])==0:
                #     a[i] = random.choice(actions)
                # else:
                #     a[i] = np.argmax(q[i][int(eta_list[i][0])][int(eta_list[i][1])][u1_swarm_list[i]])

                # if max(q[i][int(eta_list[i][0])][int(eta_list[i][1])][u1_swarm])==0:
                #     a[i] = random.choice(actions)
                # else:
                #     a[i] = np.argmax(q[i][int(eta_list[i][0])][int(eta_list[i][1])][u1_swarm])

                if max(q[i][s_x[i]][s_y[i]][u1_swarm_list[i]])==0:
                    a[i] = random.choice(actions)
                else:
                    a[i] = np.argmax(q[i][s_x[i]][s_y[i]][u1_swarm_list[i]])

                # if max(q[i][s_x[i]][s_y[i]][u1_swarm])==0:
                #     a[i] = random.choice(actions)
                # else:
                #     a[i] = np.argmax(q[i][s_x[i]][s_y[i]][u1_swarm])


                # if max(q[i].get(current_state_AS[i],[0,0,0,0]))==0:
                #     a[i] = random.choice(actions)
                #     # print('action is', a[i])
                # else:
                #     # a[i] = np.argmax(q[i][s[i]])
                #     a[i] = np.argmax(q[i].get(current_state_AS[i]))
                #     # print('ssactions isss', a[i])

        curriculum.add_step()
        
        # print('actions list is:', a)
        task.execute_action(a)

        # a = task.get_last_action() # due to MDP slip

        # u2_swarm = swarm_reward_machine.get_next_state(u1_swarm, events)
        s_new_x, s_new_y = task.get_state_vector()
        # print('s_new_x is:', s_new_x)
        # print('s_new_y is:', s_new_y)
        # s_new_2dim = state_transfer_2_dim(s_new, 8, 8, testing_params.agents_num)
 
        random_agent, random_neighbor = select_random_neighbor(a_full)
        # print(random_agent, random_neighbor)

        # Update estimation for active agent in x and y
        gm_x_estimate[random_agent] = ((gm_x_estimate[random_agent]
                                        + gm_x_estimate[random_neighbor])/2
                                        + s_new_x[random_agent]
                                        - s_x[random_agent])
        
        gm_y_estimate[random_agent] = ((gm_y_estimate[random_agent]
                                        + gm_y_estimate[random_neighbor])/2
                                        + s_new_y[random_agent]
                                        - s_y[random_agent])
        
        # Update estimation for neighbor in x and y
        gm_x_estimate[random_neighbor] = ((gm_x_estimate[random_agent]
                                    + gm_x_estimate[random_neighbor])/2
                                    + s_new_x[random_neighbor]
                                - s_x[random_neighbor])

        gm_y_estimate[random_neighbor] = ((gm_y_estimate[random_agent]
                                    + gm_y_estimate[random_neighbor])/2
                                    + s_new_y[random_neighbor]
                                - s_y[random_neighbor])         
    

        # Update estimation for the rest of the agents

        for i in range(testing_params.agents_num):
            if i in [random_agent, random_neighbor]:
                continue
            
            gm_x_estimate[i] = (gm_x_estimate[i]
                                        + s_new_x[i]
                                    - s_x[i]) 
            
            gm_y_estimate[i] = (gm_y_estimate[i]
                                        + s_new_y[i]
                                    - s_y[i]) 
        # print('GM x estimte', gm_x_estimate)
        # The events(labels) should come from estimated GMs
        
        # print('u1_swarm_list is:', u1_swarm_list)
        # print('u1_swarm_true is:', u1_swarm_true)

        events_estimate, events_true, events_own = task.get_true_propositions(gm_x_estimate,gm_y_estimate, error_upp_bound, a_full, s_x, s_y)
        # print('events_estimate is:', events_estimate)
        # print('events_true is:', events_true)
        # print('events_true is:', events_true)
        # for i in range(testing_params.agents_num):
        #     u2_swarm_estimate[i] = rm_learned.get_next_state(u1_swarm_estimate[i], events_estimate[i])

        u2_swarm_list = [rm_learned.get_next_state(u1_swarm_list[i], events_estimate[i]) for i in range(testing_params.agents_num)]
        u2_swarm_list_for_learn = [rm_learned.get_next_state(u1_swarm_list_for_learn[i], events_true) for i in range(testing_params.agents_num)]

        #u2_swarm_list = [rm_learned.get_next_state(u1_swarm_list[i], events_own[i]) for i in range(testing_params.agents_num)]

        # u2_swarm = rm_learned.get_next_state(u1_swarm, events_true)
        # u2_swarm = rm_learned.get_next_state(u1_swarm, events_own)

        # print('u1_swarm_list is:', u1_swarm_list)
        # print('u2_swarm_list is:', u2_swarm_list)

        u2_swarm_true = rm_true.get_next_state(u1_swarm_true, events_true)
       
        # print('u2_swarm_list is:', u2_swarm_list)
        # print('u2_swarm_true is:', u2_swarm_true)

        # for i in range(testing_params.agents_num):
        #     if reward[i]<=0:
        #         # reward_true[i] = rm_true.get_reward(u1_swarm_true,u2_swarm_true)

        #         # reward[i] = rm_true.get_reward(u1_swarm,u2_swarm)

        #         reward[i] = rm_true.get_reward(u1_swarm_list[i],u2_swarm_list[i])   

        for i in range(testing_params.agents_num):
            if reward[i]<=0:
                reward_true[i] = rm_true.get_reward(u1_swarm_true,u2_swarm_true)

                # reward[i] = rm_true.get_reward(u1_swarm,u2_swarm)

                reward[i] = rm_learned.get_reward(u1_swarm_list[i],u2_swarm_list[i])   
                
                
        # print('reward is:', reward)
        # print('reward_true is:', reward_true)
        # It contains one hot vector for the all the agents
        # label_one_hot_vector = get_one_hot_vector(events_estimate)
        label_one_hot_vector = get_one_hot_vector(events_true)

        next_state_AS = []
        # for i in range(num_agents):
        #     next_state_AS.append(tuple([s_new_x[i], s_new_y[i]] + label_one_hot_vector[i]))
        
        # print('current state AS after action execution is:', current_state_AS)
        # print('Is this rm terminal state:',rm_true.is_terminal_state(u2_swarm_true))
        # exit() 
        

        for i in range(num_agents):
            if update_q[i]:
                # print('updating q-values')
                neighbors_list = np.where(a_full[i] == 1)[0]

                # Add the agent itself to the list
                neighbors_list = np.append(neighbors_list, i)
                
                # print('neighbors list is:', neighbors_list)
                x_neighbors_new = np.array([s_new_x[j] for j in neighbors_list])
                mean_x_new = int(np.round(np.mean(x_neighbors_new)))

                y_neighbors_new = np.array([s_new_y[j] for j in neighbors_list])
                mean_y_new = int(np.round(np.mean(y_neighbors_new)))

                eta_list_new[i] = [mean_x_new, mean_y_new]
                # print('index is:', i)
                
                # print(eta_list)
                # print(eta_list[i][0])
                # exit()

                # q[i][int(eta_list[i][0])][int(eta_list[i][1])][u1_swarm_list[i]][a[i]] = (1 - alpha) * q[i][int(eta_list[i][0])][int(eta_list[i][1])][u1_swarm_list[i]][a[i]] + alpha * (reward[i] + gamma * np.amax(q[i][int(eta_list_new[i][0])][int(eta_list_new[i][1])][u2_swarm_list[i]]))


                # q[i][int(eta_list[i][0])][int(eta_list[i][1])][u1_swarm][a[i]] = (1 - alpha) * q[i][int(eta_list[i][0])][int(eta_list[i][1])][u1_swarm][a[i]] + alpha * (reward[i] + gamma * np.amax(q[i][int(eta_list_new[i][0])][int(eta_list_new[i][1])][u2_swarm]))

                q[i][s_x[i]][s_y[i]][u1_swarm_list[i]][a[i]] = (1 - alpha) * q[i][s_x[i]][s_y[i]][u1_swarm_list[i]][a[i]] + alpha * (reward[i] + gamma * np.amax(q[i][s_new_x[i]][s_new_y[i]][u2_swarm_list[i]]))

                # q[i][s_x[i]][s_y[i]][u1_swarm][a[i]] = (1 - alpha) * q[i][s_x[i]][s_y[i]][u1_swarm][a[i]] + alpha * (reward[i] + gamma * np.amax(q[i][s_new_x[i]][s_new_y[i]][u2_swarm]))

                # q_current_vals = q[i].get(current_state_AS[i], [0,0,0,0])
                # q_next_vals = q[i].get(next_state_AS[i], [0,0,0,0])
                # q_current_vals[a[i]] = (1 - alpha)*q_current_vals[a[i]] +  alpha * (reward[i] + gamma*np.amax((q_next_vals)))
                if reward[i] > 0:
                    # print('reward is:', reward[i])
                    # print('set q update to 0')
                    update_q[i] = 0
                
        
        # all_events.append(events)
        all_events.append(events_true)
        

        # print('before mean is', reward)
        # training_reward += np.mean(reward)
        # training_reward = np.mean(reward)
        training_reward = np.mean(reward_true[0])

        # print(f"training_reward is here:{training_reward}")

        # Printing
        if show_print and (t+1) % learning_params.print_freq == 0:
            print("Step:", t+1, "\tTotal reward:", training_reward)

        if testing_params.test and curriculum.get_current_step() % testing_params.test_freq==0:
            testing_reward = tester_learned.run_test(curriculum.get_current_step(), run_aqrm_test, rm_learned, rm_true, is_rm_learned, q, num_agents,
                                                     a_full,                  #swarm_reward_machine
                                                     )
        # print('rm_learned is', rm_learned)
        # print('rm_true is', rm_true)
        # exit()

        if is_rm_learned==0:
            if task.is_env_game_over() or rm_true.is_terminal_state(u2_swarm_true):
                break
                # # Restarting the game
                # task = Game(task_params)
                # if curriculum.stop_task(t):
                #     break
                # s2, s2_features = task.get_state_and_features()
                # u2_true = rm_true.get_initial_state()

        else:
           if task.is_env_game_over() or rm_true.is_terminal_state(u2_swarm_true):
                break
           
        # checking the steps time-out
        if curriculum.stop_learning():
            break

        # Moving to the next state
        u1_swarm_true = u2_swarm_true
        # u1_swarm = u2_swarm
        u1_swarm_list = u2_swarm_list
        u1_swarm_list_for_learn = u2_swarm_list_for_learn
        current_state_AS = next_state_AS
        s_x, s_y = s_new_x, s_new_y

    # print('after steps finished')
    # print('rm_learned state is:', u2_swarm_list[0])
    # print('is rm_learned terminal state:', rm_learned.is_terminal_state(u2_swarm_list[0]))
    # print('rm_true state is:', u2_swarm_true)
    # print('is rm_true terminal state:', rm_true.is_terminal_state(u2_swarm_true))
    # (is_rm_learned) and
    if (not rm_learned.is_terminal_state(u2_swarm_list_for_learn[0])) and (not rm_true.is_terminal_state(u2_swarm_true)):
        is_conflicting_list[0] = 0
    elif (rm_learned.is_terminal_state(u2_swarm_list_for_learn[0]) and rm_true.is_terminal_state(u2_swarm_true)):
        is_conflicting_list[0] = 0
    else:
        is_conflicting_list[0] = 1

    # print('is conflicting_list after steps finished is:', is_conflicting_list[0])
    
    step_count=t

    if testing_reward is None:
        is_test_result = 0
        testing_reward = previous_testing_reward
    else:
        is_test_result = 1

    if show_print: print("Done! Total reward:", training_reward)

    return all_events, training_reward, step_count, is_conflicting_list[0], testing_reward, is_test_result, q


def run_aqrm_test(reward_machines, task_params, rm, rm_true, is_learned, q, learning_params, testing_params, optimal, num_agents, a_full,
                #    swarm_reward_machine
                   ):
    # Initializing parameters
    task = Game(task_params, num_agents)

    alpha = 0.9
    gamma = 0.9
    w = 0
    ok = 0
    T = 100
    # print('rm is:',rm)
    # exit()
    # Starting interaction with the environment
    r_total = 0
    a = [0 for i in range(num_agents)]
    reward = [0 for i in range(num_agents)]
    
    # u1_swarm = rm.get_initial_state()
    u1_swarm_list = [rm.get_initial_state() for i in range(num_agents)]
    # u2_swarm = rm.get_initial_state()
    u1_swarm_true = rm_true.get_initial_state()

    gm_x_estimate = [0 for _ in range(testing_params.agents_num)]
    gm_y_estimate = [0 for _ in range(testing_params.agents_num)]
    error_upp_bound = 1.5
    
    # N = testing_params.agents_num
    # # n1_set = [1, 1, 2, 3]
    # # n2_set = [2, 4, 3, 4]
    # n1_set, n2_set = create_digraph_nodes_edge(N)

    

    # adjacency_matrix, lambda_2, a_full = get_topology_matrix(N, n1_set ,n2_set)
    
    s_x, s_y = task.get_state_vector()
    estimate_label, true_label, events_own = task.get_true_propositions(gm_x_estimate, gm_y_estimate, error_upp_bound, a_full, s_x, s_y) 
    label_one_hot_vector = get_one_hot_vector(estimate_label)
    current_state_AS = []
    for i in range(num_agents):
        current_state_AS.append(tuple([s_x[i], s_y[i]] + label_one_hot_vector[i]))

    # print('current state AS before start in testing is:', current_state_AS)
    for t in range(testing_params.num_steps):

        # Choosing an action to perform
        actions = task.get_actions()
        # s_x, s_y = task.get_state_vector()
        
        # s_old = state_transfer_2_dim(s, 8, 8, testing_params.agents_num)
        eta_list = np.zeros([num_agents, 2])

        for i in range(num_agents):
            # if max(q[i][s[i]][u1]) == 0:
            
            # if max(q[i][s[i]]) == 0:

            neighbors_list = np.where(a_full[i] == 1)[0]

            # Add the agent itself to the list
            neighbors_list = np.append(neighbors_list, i)

            # Get x coordinates of neighbors and average them
            x_neighbors = np.array([s_x[j] for j in neighbors_list])
            mean_x = int(np.round(np.mean(x_neighbors)))

            # Get y coordinates of neighbors and average them
            y_neighbors = np.array([s_y[j] for j in neighbors_list])
            mean_y = int(np.round(np.mean(y_neighbors)))

            # Store results
            eta_list[i] = [mean_x, mean_y]

            
            # if max(q[i][int(eta_list[i][0])][int(eta_list[i][1])][u1_swarm_list[i]])==0:
            #     a[i] = random.choice(actions)
            # else:
            #     a[i] = np.argmax(q[i][int(eta_list[i][0])][int(eta_list[i][1])][u1_swarm_list[i]])

            # if max(q[i][int(eta_list[i][0])][int(eta_list[i][1])][u1_swarm])==0:
            #     a[i] = random.choice(actions)
            # else:
            #     a[i] = np.argmax(q[i][int(eta_list[i][0])][int(eta_list[i][1])][u1_swarm])

            if max(q[i][s_x[i]][s_y[i]][u1_swarm_list[i]])==0:
                a[i] = random.choice(actions)
            else:
                a[i] = np.argmax(q[i][s_x[i]][s_y[i]][u1_swarm_list[i]])

            # if max(q[i][s_x[i]][s_y[i]][u1_swarm])==0:
            #     a[i] = random.choice(actions)
            # else:
            #     a[i] = np.argmax(q[i][s_x[i]][s_y[i]][u1_swarm])
        
            # if max(q[i].get(current_state_AS[i],[0,0,0,0]))==0:
            #         a[i] = random.choice(actions)
            #         # print('action is', a[i])
            # else:
            #     # a[i] = np.argmax(q[i][s[i]])
            #     a[i] = np.argmax(q[i][current_state_AS[i]])
            #     # print('ssactions isss', a[i])

        # print('actions list in testing is', a)
        task.execute_action(a)
        # a = task.get_last_action() # due to MDP slip
        

        
        # s_new_2dim = state_transfer_2_dim(s_new, 8, 8, testing_params.agents_num)
        s_new_x, s_new_y = task.get_state_vector()

        random_agent, random_neighbor = select_random_neighbor(a_full)
        # print(random_agent, random_neighbor)

        # Update estimation for active agent in x and y
        gm_x_estimate[random_agent] = ((gm_x_estimate[random_agent]
                                        + gm_x_estimate[random_neighbor])/2
                                        + s_new_x[random_agent]
                                        - s_x[random_agent])
        
        gm_y_estimate[random_agent] = ((gm_y_estimate[random_agent]
                                        + gm_y_estimate[random_neighbor])/2
                                        + s_new_y[random_agent]
                                        - s_y[random_agent])
        
        # Update estimation for neighbor in x and y
        gm_x_estimate[random_neighbor] = ((gm_x_estimate[random_agent]
                                    + gm_x_estimate[random_neighbor])/2
                                    + s_new_x[random_neighbor]
                                - s_x[random_neighbor])

        gm_y_estimate[random_neighbor] = ((gm_y_estimate[random_agent]
                                    + gm_y_estimate[random_neighbor])/2
                                    + s_new_y[random_neighbor]
                                - s_y[random_neighbor])        
    

        # Update estimation for the rest of the agents
        for i in range(testing_params.agents_num):
            if i in [random_agent, random_neighbor]:
                continue
            
            gm_x_estimate[i] = (gm_x_estimate[i]
                                        + s_new_x[i]
                                    - s_x[i]) 
            
            gm_y_estimate[i] = (gm_y_estimate[i]
                                        + s_new_y[i]
                                    - s_y[i]) 
        # print('GM x estimte', gm_x_estimate)


        events_estimate, events_true, events_own = task.get_true_propositions(gm_x_estimate,gm_y_estimate, error_upp_bound, a_full, s_x, s_y)

        # print('events_true in testing is:', events_true)

        #for i in range(testing_params.agents_num):
        #     # u2_swarm[i] = swarm_reward_machine.get_next_state(u1_swarm[i], event[i])
        #     u2_swarm_estimate[i] = rm.get_next_state(u1_swarm_estimate[i], events_estimate[i])

        # u2_swarm= rm.get_next_state(u1_swarm, events_true)
        # u2_swarm = rm.get_next_state(u1_swarm, events_own)

        u2_swarm_list = [rm.get_next_state(u1_swarm_list[i], events_estimate[i]) for i in range(testing_params.agents_num)]

        #u2_swarm_list = [rm.get_next_state(u1_swarm_list[i], events_own[i]) for i in range(testing_params.agents_num)]

        u2_swarm_true = rm_true.get_next_state(u1_swarm_true, events_true)
        if u2_swarm_true!=u1_swarm_true:
            print(f"U1_true: {u1_swarm_true}; U2_true: {u2_swarm_true}; Event_true: {events_true}; Steps: {t}")

        for i in range(testing_params.agents_num):
            if reward[i]<=0:
                # reward[i] = swarm_reward_machine.get_reward(u1_swarm[i],u2_swarm[i],s[i],a[i],s_new[i])
                reward[i] = rm_true.get_reward(u1_swarm_list[i],u2_swarm_list[i])   

                #reward[i] = rm_true.get_reward(u1_swarm_true,u2_swarm_true)
                # reward[i] = rm_true.get_reward(u1_swarm,u2_swarm)
                reward_swarm = rm_true.get_reward(u1_swarm_true,u2_swarm_true)
                # print('reward is here', reward[i])

        # print('reward in testing is:', reward)

        label_one_hot_vector = get_one_hot_vector(events_estimate)
        next_state_AS = []
        for i in range(num_agents):
            next_state_AS.append(tuple([s_new_x[i], s_new_y[i]] + label_one_hot_vector[i]))

        # print('current state AS after action execution in testing is:', current_state_AS)
        # r_total += 1 * learning_params.gamma**t # used in original graphing framework
        # print('Is this terminal state?', rm_true.is_terminal_state(u2_swarm_true))
        # Restarting the environment (Game Over)
        if task.is_env_game_over() or rm_true.is_terminal_state(u2_swarm_true):
            break

        # Moving to the next state
        u1_swarm_true = u2_swarm_true 
        # u1_swarm = u2_swarm # u1_swarm = u2_swarm
        u1_swarm_list = u2_swarm_list
        current_state_AS = next_state_AS # current_state_AS = next_state_AS
        s_x, s_y = s_new_x, s_new_y

    # reward_sum = 0
    # for i in range(testing_params.agents_num):
    #     print('reward for agent is', reward[i])
    #     reward_sum += reward[i]
    #reward_sum = np.mean(reward)
    reward_sum =reward_swarm
    # print(f"In testing: {reward_sum}")
    # reward_sum = trainin

    # if rm_true.is_terminal_state(u2_true) and r>0:
    #     return 1
    # else:
    #     return 0
    # if reward_sum>0:
        # print(reward_sum)
        # exit()
        

    return reward_sum

def _remove_files_from_folder(relative_path):

    dirname = os.path.abspath(os.path.dirname(__file__))


    parent_folder = os.path.normpath(os.path.join(dirname, relative_path))

    if os.path.isdir(parent_folder):
        for filename in os.listdir(parent_folder):
            absPath = os.path.join(parent_folder, filename)
            subprocess.run(["rm", absPath])
    else:
        print("There is no directory {}".format(parent_folder))


def run_aqrm_experiments(alg_name, tester, tester_learned, curriculum, num_times, show_print, show_plots, al_alg_name, sat_alg_name, pysat_hints=None):
    alg_name = alg_name.lower()
    al_alg_name = al_alg_name.lower()
    sat_alg_name = sat_alg_name.lower()

    testing_params = tester_learned.testing_params
    learning_params = tester_learned.learning_params

    num_agents = testing_params.agents_num

    file_pattern = 'swarm_rewards'
    csv_file_name = f'{file_pattern}.csv'
    
    if os.path.exists(csv_file_name):
            # If the file exists, delete it
            os.remove(csv_file_name)

    algorithm_name = alg_name
    if alg_name=="jirp":
        algorithm_name += al_alg_name
        if al_alg_name=="pysat":
            algorithm_name += sat_alg_name
        if pysat_hints is not None:
            algorithm_name += ":" + ":".join(hint.lower() for hint in pysat_hints)

    for character in tester.world.tasks[0]:
        if str.isdigit(character):
            task_id = character

    run_name = f"{tester.game_type}{task_id}{algorithm_name}"

    details_filename = f"../plotdata/details_{run_name}.csv"
    with open(details_filename, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["RUN_PARAMETERS:"])
        wr.writerow(["world:",         tester.game_type])
        wr.writerow(["  task:",        task_id])
        wr.writerow(["algorithm:",     alg_name])
        wr.writerow(["  enter_loop:",  learning_params.enter_loop])
        wr.writerow(["  step_unit:",   testing_params.num_steps])
        wr.writerow(["  total_steps:", curriculum.total_steps])
        wr.writerow(["  epsilon:",     0.30]) # WARNING: it's hard-coded in `run_aqrm_task()`
        wr.writerow(["al_algorithm:", al_alg_name.upper()])
        if "PYSAT" in al_alg_name.upper():
            wr.writerow(["  sat_algorithm:", sat_alg_name])
            if pysat_hints is None:
                wr.writerow(["  hint_at:", "never"])
            else:
                wr.writerow(["  hint_at:", "relearn"]) # WARNING: it's hard-coded (see `hm_file`)
                for hint in pysat_hints:
                    wr.writerow(["    hint:", hint.lower() if hint else "∅"])



    # time_start = time.clock()

    # just in case, delete all temporary files
    dirname = os.path.abspath(os.path.dirname(__file__))
    _remove_files_from_folder("../automata_learning_utils/data")

    # Running the tasks 'num_times'
    time_init = time.time()
    plot_dict = dict()
    rewards_plot = list()

    N = tester_learned.testing_params.agents_num
    n1_set, n2_set = create_digraph_nodes_edge(N)
    adjacency_matrix, lambda_2, a_full = get_topology_matrix(N, n1_set ,n2_set)

    # hints
    hint_dfas = None
    if pysat_hints is not None:
        # hint_dfas = [al_utils.gen_sup_hint_dfa(symbols) for symbols in pysat_hints]
        hint_dfas = list(itertools.chain.from_iterable(
            al_utils.gen_hints(symbols) for symbols in pysat_hints
        ))

    new_traces = Traces(set(), set())

    if isinstance(num_times, int):
        num_times = range(num_times)
    elif isinstance(num_times, tuple):
        num_times = range(*num_times)
    for t_i,t in enumerate(num_times):

        LIVETESTER = LiveTester(curriculum,
            show = (len(num_times)<=1),
            keep_open = True, # TODO remove
            # keep_open = show_plots,
            label = f"{run_name} - iteration {t} ({t_i+1}/{len(num_times)})",
            filebasename = run_name,
        ).start()
        task_timer = Timer()
        al_data = {
            "step": [],
            "pos": [],
            "neg": [],
            "time": [],
        }

        # Setting the random seed to 't'

        random.seed(t)
        open('./automata_learning_utils/data/data.txt','w').close
        open('./automata_learning_utils/data/automaton.txt','w').close


        # Reseting default values
        curriculum.restart()

        num_episodes = 0
        total = 0
        learned = 0
        step = 0
        enter_loop = 1
        num_conflicting_since_learn = 0
        update_rm = 0
        refreshed = 0
        testing_step = 0
        LIVETESTER.add_bool(step, 'learned', learned)

        # computing rm
        LIVETESTER.add_event(step, 'rm_update', force_update=show_plots)
        hm_file        = './automata_learning_utils/data/rm0.txt'
        hm_file_update = './automata_learning_utils/data/rm.txt'
        shutil.copy('./automata_learning/hypothesis_machine.txt', hm_file)
        # if hint_dfas is not None:
        #     print("Initial reward machine...")
        #     rm0 = al_utils.gen_dfa_from_hints(sup_hints=hint_dfas, show_plots=show_plots) # hint since BEGINNING
        #     # rm0 = al_utils.gen_partial_hint_dfa(pysat_hints[0], show_plots=show_plots) # begin with PARTIAL hint
        #     # rm0 = al_utils.gen_empty_hint_dfa(pysat_hints[0], show_plots=show_plots) # INITEMPTY
        #     rm0.export_as_reward_automaton(hm_file)
        #     Traces.rm_trace_to_symbol(hm_file)
        #     Traces.fix_rmfiles(hm_file)
        shutil.copy(hm_file, hm_file_update)


        # Creating policy bank
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()), testing_params.agents_num)
        num_features = len(task_aux.get_features())
        num_actions  = len(task_aux.get_actions())
        # q = np.zeros([1681,15,4])
        # num_agents = 100
        # num_agents = testing_params.agents_num
        # q = np.zeros([num_agents,1681,15,num_actions])
        ##############################################################
        # q = np.zeros([num_agents,64,num_actions])
        q = create_list_of_dicts(testing_params.agents_num)
        # print('q-values are:',q)
        # exit()
        ##############################################################

        hypothesis_machine = tester.get_hypothesis_machine()
        tester_learned.update_hypothesis_machine_file(hm_file)
        tester_learned.update_hypothesis_machine()
        # LIVETESTER.add_event(step, 'rm_update') # already added

        # Task loop
        automata_history = []
        rewards = list()
        episodes = list()
        steps = list()
        testing_reward = 0 #initializes value
        all_traces = Traces(set(),set())
        LIVETESTER.add_traces_size(step, all_traces, 'all_traces')
        LIVETESTER.add_traces_size(step, new_traces, 'new_traces')
        epsilon = 0.3
        tt=t+1
        print("run index:", +tt)

        # swarm_reward_machine = RewardMachine("automata_learning/swarm_reward_machine.txt")

        while not curriculum.stop_learning():
            num_episodes += 1
            # print('episode number is:', num_episodes)

            if show_print: print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
            rm_file_truth = '../experiments/craft/reward_machines/t1.txt'

            # Running 'task_rm_id' for one episode

            if learned==0:
                rm_file_learned = hm_file
                if update_rm:
                    update_rm = 0
                    refreshed = 1
                    tester_learned.update_hypothesis_machine_file(hm_file)
                    tester_learned.update_hypothesis_machine()
                    LIVETESTER.add_event(step, 'rm_refresh')
                    all_traces = Traces(set(),set())
                    LIVETESTER.add_traces_size(step, all_traces, 'all_traces')
                    num_conflicting_since_learn = 0
                    # q = np.zeros([num_agents,1681,15,4])
                    ##############################################################
                    # q = np.zeros([num_agents,64,num_actions])
                    q = create_list_of_dicts(testing_params.agents_num)
                    ##############################################################
                    enter_loop = 1
            elif update_rm:
                rm_file_learned = hm_file_update

                task_aux = Game(tester.get_task_params(curriculum.get_current_task()), num_agents)
                num_features = len(task_aux.get_features())
                num_actions = len(task_aux.get_actions())
                rm_learned = tester_learned.get_hypothesis_machine() # used to be rm_learned = tester_learned.get_reward_machines()[0]
                if len(rm_learned.U)<16:
                    print("number of states:" + str(len(rm_learned.U)))
                else:
                    update_rm = 0
                    refreshed = 1
                    tester_learned.update_hypothesis_machine_file(hm_file)
                    tester_learned.update_hypothesis_machine()
                    LIVETESTER.add_event(step, 'rm_refresh')
                    all_traces = Traces(set(), set())
                    LIVETESTER.add_traces_size(step, all_traces, 'all_traces')
                    num_conflicting_since_learn = 0
                    # q = np.zeros([num_agents,1681, 15, 4])
                    ##############################################################
                    # q = np.zeros([num_agents,64,num_actions])
                    q = create_list_of_dicts(testing_params.agents_num)
                    ##############################################################
                    enter_loop = 1
                    learned = 0
                    LIVETESTER.add_bool(step, 'learned', learned)

                update_rm = 0

            else:
                pass
            automata_history.append(rm_file_learned) #####fix this

            epsilon = epsilon*0.99

            task_timer.resume()
            all_events, found_reward, stepcount, conflicting, testing_reward, is_test, q = run_aqrm_task(
                epsilon, rm_file_truth, rm_file_learned, tester, tester_learned, curriculum, show_print, learned, step, testing_reward, q, testing_params.agents_num, a_full,
                #swarm_reward_machine
            )
            task_timer.stop()
            LIVETESTER.add_bool(step, 'conflicting', conflicting)
            LIVETESTER.add_bool(step, 'is_positive', found_reward>0)
            LIVETESTER.add_bool(step, 'is_test', is_test)
            # print(",".join(all_events), "\n") #################################################


            #set up traces; we remove anything foreign to our ground truth formula

            # if tester.game_type=="officeworld":
            #     while 'h' in all_events:
            #        all_events.remove('h')
            # elif tester.game_type=="trafficworld":
            #     while 'f' in all_events:
            #        all_events.remove('f')
            #     while 'g' in all_events:
            #        all_events.remove('g')
            # elif tester.game_type=="craftworld":
            #     while 'd' in all_events:
            #        all_events.remove('d')
            #     while 'g' in all_events:
            #        all_events.remove('g')
            #     while 'h' in all_events:
            #        all_events.remove('h')

            while '' in all_events:
                all_events.remove('')
            if (conflicting==1 or refreshed==1):
                all_traces.add_trace(all_events, found_reward, learned)
                LIVETESTER.add_traces_size(step, all_traces, 'all_traces')

            if (num_episodes%100==0):
                print("run index:", +tt)
                toprint = "Total training reward at "+str(step)+": "+str(total)
                print(toprint)

            if num_episodes>5000:
                num_episodes

            total += found_reward
            step += stepcount
            num_conflicting_since_learn += conflicting
            rewards.append(found_reward)
            episodes.append(num_episodes)
            steps.append(step)

            if is_test:
                testing_step += testing_params.test_freq
                plot_dict.setdefault(testing_step, [])
                plot_dict[testing_step].append(testing_reward)
                LIVETESTER.add_reward(testing_step, testing_reward)
                # plot_dict[testing_step].append(found_reward)
                # LIVETESTER.add_reward(testing_step, found_reward)


            if learned==1:

                if num_episodes%learning_params.relearn_period==0 and (num_conflicting_since_learn>0):
                    enter_loop = 1

                if conflicting==1:
                    new_traces.add_trace(all_events, found_reward, learned)
                    LIVETESTER.add_traces_size(step, new_traces, 'new_traces')



            # if enter_loop:
            #     print("\x1B[1;31;44m enter loop (%d positives) \x1B[m" % len(all_traces.positive))
            # enter_loop = 0
            if (len(all_traces.positive)<learning_params.enter_loop) and enter_loop:
                LIVETESTER.add_event(step, 'rm_learn_failed', force_update=show_plots)
            if (len(all_traces.positive)>=learning_params.enter_loop) and enter_loop:
                LIVETESTER.add_event(step, 'rm_learn', force_update=show_plots)

                # positive = set()
                # negative = set()
                #
                # if learned==0:
                #     if len(all_traces.positive)>0:
                #         for i in list(all_traces.positive):
                #             if all_traces.symbol_to_trace(i) not in positive:
                #                 positive.add(all_traces.symbol_to_trace(i))
                #     if len(all_traces.negative)>0:
                #         for i in list(all_traces.negative):
                #             if all_traces.symbol_to_trace(i) not in negative:
                #                 negative.add(all_traces.symbol_to_trace(i))
                # else:
                #     if len(new_traces.positive)>0:
                #         for i in list(new_traces.positive):
                #             if new_traces.symbol_to_trace(i) not in positive:
                #                 positive.add(new_traces.symbol_to_trace(i))
                #     if len(new_traces.negative)>0 and len(all_traces.negative):
                #         for i in list(new_traces.negative):
                #             if new_traces.symbol_to_trace(i) not in negative:
                #                 negative.add(new_traces.symbol_to_trace(i))
                """equivalent:"""
                traces = all_traces if not learned else new_traces
                positive = set(Traces.symbol_to_trace(i) for i in traces.positive)
                negative = set(Traces.symbol_to_trace(i) for i in traces.negative)

                # print("PPP", positive) ####################################""
                # print("NNN", negative)


                positive_new = set() ## to get rid of redundant prefixes
                negative_new = set()

                if not learned:
                    for ptrace in positive:
                        new_trace = list()
                        previous_prefix = None #arbitrary
                        for prefix in ptrace:
                            if prefix != previous_prefix:
                                new_trace.append(prefix)
                            previous_prefix = prefix
                        positive_new.add(tuple(new_trace))

                    for ntrace in negative:
                        new_trace = list()
                        previous_prefix = None #arbitrary
                        for prefix in ntrace:
                            if prefix != previous_prefix:
                                new_trace.append(prefix)
                            previous_prefix = prefix
                        negative_new.add(tuple(new_trace))
                    if tester.game_type=="trafficworld":
                        if len(negative_new)<50:
                            negative_to_store = negative_new
                        else:
                            negative_to_store = set(random.sample(negative_new, 50))
                    else:
                        negative_to_store = negative_new
                    positive_to_store = positive_new
                    negative_new = negative_to_store
                    positive_new = positive_to_store

                    negative = set()
                    positive = set()

                else:
                    for ptrace in positive:
                        new_trace = list()
                        for prefix in ptrace:
                            new_trace.append(prefix)
                        positive_to_store.add(tuple(new_trace))
                        positive_new = positive_to_store
                        negative_new = negative_to_store

                    for ntrace in negative:
                        new_trace = list()
                        for prefix in ntrace:
                            new_trace.append(prefix)
                        negative_to_store.add(tuple(new_trace))
                        positive_new = positive_to_store
                        negative_new = negative_to_store

                traces_numerical = Traces(positive_new, negative_new)
                traces_file = './automata_learning_utils/data/data.txt'
                traces_numerical.export_traces(traces_file)
                LIVETESTER.add_traces_size(step, traces_numerical, 'traces_numerical')

                if learned == 1:
                    shutil.copy('./automata_learning_utils/data/rm.txt', '../experiments/use_past/t2.txt')

                al_utils.al_timer.reset()
                automaton_visualization_filename = al_utils.learn_automaton(traces_file, show_plots,
                    automaton_learning_algorithm=al_alg_name,
                    pysat_algorithm=sat_alg_name,
                    sup_hint_dfas=hint_dfas,
                    output_reward_machine_filename=hm_file_update,
                )
                al_data["step"].append(step)
                al_data["pos"].append(len(traces_numerical.positive))
                al_data["neg"].append(len(traces_numerical.negative))
                al_data["time"].append(al_utils.al_timer.elapsed())
                # if al_utils.al_timer.elapsed() > 10: #TODO REMOVE
                #     shutil.copy(traces_file, '../plotdata/data{:d}{:d}_{}_{:02d}{:02d}.txt'.format(
                #         2, int(task_id),
                #         algorithm_name[4:].upper(),
                #         t, len(al_data["time"])-1,
                #     ))

                # t2 is previous, t1 is new
                Traces.rm_trace_to_symbol(hm_file_update)
                Traces.fix_rmfiles(hm_file_update)

                if learned == 0:
                    shutil.copy('./automata_learning_utils/data/rm.txt',
                                             '../experiments/use_past/t2.txt')

                tester_learned.update_hypothesis_machine_file(hm_file_update) ## NOTE WHICH TESTER IS USED
                tester_learned.update_hypothesis_machine()
                # LIVETESTER.add_event(step, 'rm_learn') # already added


                print("learning")
                parent_path = os.path.abspath("../experiments/use_past/")
                os.makedirs(parent_path, exist_ok=True)

                shutil.copy(hm_file_update, '../experiments/use_past/t1.txt')
                if tester.game_type == 'officeworld':
                    current_and_previous_rms = '../experiments/office/tests/use_previous_experience.txt'
                elif tester.game_type == 'craftworld':
                    current_and_previous_rms = '../experiments/craft/tests/use_previous_experience.txt'
                elif tester.game_type == 'trafficworld':
                    current_and_previous_rms = '../experiments/traffic/tests/use_previous_experience.txt'
                elif tester.game_type == 'taxiworld':
                    current_and_previous_rms = '../experiments/taxi/tests/use_previous_experience.txt'
                else:
                    raise NotImplementedError(tester.game_type)


                tester_current = Tester(learning_params,testing_params,current_and_previous_rms)




                learned = 1
                LIVETESTER.add_bool(step, 'learned', learned)
                enter_loop = 0
                num_conflicting_since_learn = 0
                update_rm = 1

            if num_episodes%learning_params.relearn_period==0:
                new_traces = Traces(set(), set())
                LIVETESTER.add_traces_size(step, new_traces, 'new_traces')

            # if (learned==1 and num_episodes==1000):
            #
            #     tester_learned.update_hypothesis_machine()
            #     LIVETESTER.add_event(step, 'rm_update')
            #
            #
            #
            #     shutil.copy(hm_file_update, '../experiments/use_past/t2.txt')
            #     if tester.game_type == 'officeworld':
            #         current_and_previous_rms = '../experiments/office/tests/use_previous_experience.txt'
            #     elif tester.game_type == 'craftworld':
            #         current_and_previous_rms = '../experiments/craft/tests/use_previous_experience.txt'
            #     else:
            #         current_and_previous_rms = '../experiments/traffic/tests/use_previous_experience.txt'
            #
            #
            #     tester_current = Tester(learning_params,testing_params,current_and_previous_rms)
            #
            #
            #     q_old = np.copy(q)
            #     for ui in range(len(tester_current.reward_machines[0].get_states())):
            #         if not tester_current.reward_machines[0]._is_terminal(ui):
            #             is_transferred = 0
            #             for uj in range(len(tester_current.reward_machines[1].get_states())):
            #                 if not tester_current.reward_machines[1]._is_terminal(uj):
            #                     if tester_current.reward_machines[0].is_this_machine_equivalent(ui,tester_current.reward_machines[1],uj):
            #                         for s in range(len(q)):
            #                             if sum(q_old[s][uj])>0:
            #                                 q_old
            #                             q[s][ui] = np.copy(q_old[s][uj])
            #                         is_transferred = 1
            #                     else:
            #                         if not is_transferred:
            #                             for s in range(len(q)):
            #                                 q[s][ui] = 0
            #     # for ui in range(len(tester_current.reward_machines[0].get_states())):
            #     #     for s in range(len(q)):
            #     #         q[s][ui] = 0


        # Backing up the results
        print('Finished iteration ',t)
        reward_list = []
        reward_step = None # first step at which G(reward=1)
        for step,rwds in plot_dict.items():
            reward = rwds[-1]
            reward_list.append(reward)

            if not reward:
                reward_step = None
            elif reward_step is None:
                reward_step = step

        with open(details_filename, 'a') as f:
            wr = csv.writer(f)
            wr.writerow(["ITERATION_DETAIL:", t])
            wr.writerow(["task_time:", task_timer.elapsed()])
            wr.writerow(["al_step:", *al_data["step"]])
            wr.writerow(["al_pos:",  *al_data["pos"]])
            wr.writerow(["al_neg:",  *al_data["neg"]])
            wr.writerow(["al_time:", *al_data["time"]])
            wr.writerow(["total_time:", sum((task_timer.elapsed(),*al_data["time"]))])
            wr.writerow(["reward_step:", reward_step])

        LIVETESTER.close()


        if isinstance(reward_list[0], int):
            reward_list = [reward_list]

        total_rewards_list = reward_list
        transposed_rewards = list(zip(*total_rewards_list))

        # print('transposed_rewards is:', transposed_rewards)

        if os.path.exists(csv_file_name):
            df = pd.read_csv(csv_file_name)
            headers = []
            for i in range(len(total_rewards_list)):
                headers.append(f'Iteration_{t_i}_Agent_{i+1}_Reward')
            df[headers] = transposed_rewards
            df.to_csv(csv_file_name, index=False)
        else:
            headers = []
            for i in range(len(total_rewards_list)):
                headers.append(f'Iteration_{t_i}_Agent_{i+1}_Reward')
            df = pd.DataFrame(transposed_rewards, columns=headers)
            df.to_csv(csv_file_name, index=False)
    # Showing results

    prc_25 = list()
    prc_50 = list()
    prc_75 = list()


    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()
    steps_plot = list()

    for step in plot_dict.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))

        rewards_plot.append(sum(plot_dict[step])/len(plot_dict[step]))
        prc_25.append(sum(current_25)/len(current_25))
        prc_50.append(sum(current_50)/len(current_50))
        prc_75.append(sum(current_75)/len(current_75))
        steps_plot.append(step)



    # tester.plot_performance(steps_plot,prc_25,prc_50,prc_75) #TODO: uncomment
    # tester.plot_this(steps_plot,rewards_plot) #TODO: uncomment

    output_filename = f"../plotdata/{run_name}.csv"

    with open(output_filename, 'w') as f:
        wr = csv.writer(f)
        wr.writerows(list(plot_dict.values()))


    avg_filename = f"../plotdata/avgreward_{run_name}.txt"

    with open(avg_filename, 'w') as f:
        f.write("%s\n" % str(sum(rewards_plot) / len(rewards_plot)))
        for item in rewards_plot:
            f.write("%s\n" % item)
