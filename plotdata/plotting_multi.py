import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def extract_agent_rewards(df, agent_ind, iters_num):
    agent_iter_list = [
        df[f"Iteration_{iter_ind}_Agent_{agent_ind}_Reward"]
        for iter_ind in range(iters_num)
    ]
    return agent_iter_list

def calculate_moving_average(rewards, window_size):
    return rewards.rolling(window=window_size, min_periods=1).mean()

def calculate_percentiles(data, percentiles):
    return {p: np.percentile(data, p, axis=0) for p in percentiles}

def plot_rewards(E, percentiles, label, color, linestyle='-'):
    plt.plot(E, percentiles[50], linewidth=2, color=color, label=label, linestyle=linestyle)
    plt.fill_between(E, percentiles[25], percentiles[75], alpha=0.3, color=color)

def process_and_plot(file_info, window_size, agent_ind, percentiles_to_calculate,
    iters_num = None
                     ):
    percentiles_data = []

    for info in file_info:
        df = load_data(info['file_path'])
        if iters_num is None:
            iters_num = df.filter(regex=f"Iteration_.*_Agent_{agent_ind}_Reward").shape[1]
        
        agent_rewards = extract_agent_rewards(df, agent_ind, iters_num)
        agent_moving_avg = np.array([calculate_moving_average(rewards, window_size) for rewards in agent_rewards])
        agent_percentiles = calculate_percentiles(agent_moving_avg, percentiles_to_calculate)
        
        E = np.arange(agent_moving_avg.shape[1])
        percentiles_data.append((E, agent_percentiles, info['label'], info['color'], info.get('linestyle', '-')))

    plt.figure(figsize=(12, 6))

    for E, agent_percentiles, label, color, linestyle in percentiles_data:
        plot_rewards(E, agent_percentiles, label, color, linestyle)

    plt.grid()
    plt.xlabel('Episode', fontsize=22)
    plt.ylabel('Discounted Cumulative Reward', fontsize=22)
    plt.title('Case Study 1', fontsize=22)
    plt.xticks(fontsize=28)  # Set font size for x-axis ticks
    plt.yticks(fontsize=28) 
    plt.xlim(0, 2000)
    plt.legend(fontsize=24 , bbox_to_anchor=(0.55, 0.94))
    plt.savefig('case_1_reward.png', format="png", bbox_inches='tight', dpi=300)
    plt.show()


def main():
    

    file_info = [
        {'file_path': './swarm_rewards_c1_abc_5run_seed0.csv',
          'label': 'SwaRM-L', 'color': 'blue',
           # 'linestyle': '--'
           },
        {'file_path': './qas_c1_ab_5run_seed0.csv', 'label': 'QAS', 'color': 'red'},
          {'file_path': './ddqn_c1_ab_5run_seed0.csv', 'label': 'DDQN', 'color': 'green'},
        #   {'file_path': './swarm_rewards_ab_RM_Given_c1.csv', 'label': 'LAR-swaRM', 'color': 'cyan'},
        #   {'file_path': './hrl_c1_seed0_5runs.csv', 'label': 'HRL', 'color': 'k'},
          {'file_path': './hrl_c1_seed0_5run_stp50.csv', 'label': 'HRL', 'color': 'k'}
        
    ]



    window_size = 20
    agent_ind = 1
    percentiles_to_calculate = [25, 50, 75]

    process_and_plot(file_info, window_size, 
                     agent_ind, percentiles_to_calculate,
                       iters_num=5)

if __name__ == "__main__":
    main()

