# The office environment

This environment consists of one fix grid map (which is shown in the paper) that contains the sub-tasks required for the swarm to perform.
We encode the tasks using the reward machine states and labels where 
labels trigger the transition from current reward machine state to the
next reward machine state. 
Labels become true when swarm GMs reach the desired values marking that 
subtask complete thus, encoding the task for the swarm.