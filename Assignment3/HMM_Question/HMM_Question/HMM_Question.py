import numpy as np
import matplotlib.pyplot as plt
import random, os
from tqdm import tqdm
from roomba_class import Roomba
import csv


# ### Setup Environment

def seed_everything(seed: int):
    """Seed everything for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def is_obstacle(position):
    """Check if the position is outside the grid boundaries (acting as obstacles)."""
    x, y = position
    return x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT

def setup_environment(seed=111):
    """Setup function for grid and direction definitions."""
    global GRID_WIDTH, GRID_HEIGHT, HEADINGS, MOVEMENTS
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    HEADINGS = ['N', 'E', 'S', 'W']
    MOVEMENTS = {
        'N': (0, -1),
        'E': (1, 0),
        'S': (0, 1),
        'W': (-1, 0),
    }
    print("Environment setup complete with a grid of size {}x{}.".format(GRID_WIDTH, GRID_HEIGHT))
    seed_everything(seed)
    return GRID_WIDTH, GRID_HEIGHT, HEADINGS, MOVEMENTS


# ### Sensor Movements

def simulate_roomba(T, movement_policy,sigma):
    """
    Simulate the movement of a Roomba robot for T time steps and generate noisy observations.

    Parameters:
    - T (int): The number of time steps for which to simulate the Roomba's movement.
    - movement_policy (str): The movement policy dictating how the Roomba moves.
                             Options may include 'straight_until_obstacle' or 'random_walk'.
    - sigma (float): The standard deviation of the Gaussian noise added to the true position
                     to generate noisy observations.

    Returns:
    - tuple: A tuple containing three elements:
        1. true_positions (list of tuples): A list of the true positions of the Roomba
                                            at each time step as (x, y) coordinates.
        2. headings (list): A list of headings of the Roomba at each time step.
        3. observations (list of tuples): A list of observed positions with added Gaussian noise,
                                          each as (obs_x, obs_y).
    """
    # Start at the center
    start_pos = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
    start_heading = random.choice(HEADINGS)
    roomba = Roomba(MOVEMENTS, HEADINGS,is_obstacle,start_pos, start_heading, movement_policy)

    true_positions = []
    observations = []
    headings = []

    print(f"Simulating Roomba movement for policy: {movement_policy}")
    for _ in tqdm(range(T), desc="Simulating Movement"):
        position = roomba.move()
        heading = roomba.heading
        true_positions.append(position)
        headings.append(heading)

        # Generate noisy observation
        noise = np.random.normal(0, sigma, 2)
        observed_position = (position[0] + noise[0], position[1] + noise[1])
        observations.append(observed_position)

    return true_positions, headings, observations


# ### Implement Functions

def emission_probability(state, observation, sigma):
    """
    Calculate the emission probability in log form for a given state and observation using a Gaussian distribution.

    Parameters:
    - state (tuple): The current state represented as (position, heading),
                     where position is a tuple of (x, y) coordinates.
    - observation (tuple): The observed position as a tuple (obs_x, obs_y).
    - sigma (float): The standard deviation of the Gaussian distribution representing observation noise.

    Returns:
    - float: The log probability of observing the given observation from the specified state.
    """
    ###### YOUR CODE HERE ######
    #pass
    (x, y) = state[0]
    a = observation[0]
    b = observation[1]
    d = (a - x) ** 2 + (b - y) ** 2
    probability = -d / (2 * sigma ** 2) - np.log(2 * np.pi * sigma ** 2)
    return probability

def transition_probability(prev_state, curr_state, movement_policy):
    """
    Calculate the transition probability in log form between two states based on a given movement policy.

    Parameters:
    - prev_state (tuple): The previous state represented as (position, heading),
                          where position is a tuple of (x, y) coordinates and heading is a direction.
    - curr_state (tuple): The current state represented as (position, heading),
                          similar to prev_state.
    - movement_policy (str): The movement policy that dictates how transitions are made. 
                             Options are 'straight_until_obstacle' and 'random_walk'.

    Returns:
    - float: The log probability of transitioning from prev_state to curr_state given the movement policy.
             Returns 0.0 (log(1)) for certain transitions, -inf (log(0)) for impossible transitions,
             and a uniform log probability for equal transitions in the case of random walk.
    """
    ###### YOUR CODE HERE ######
    #pass
    previousPosition = prev_state[0]
    currentPosition = curr_state[0]

    previousHeading = prev_state[1]
    currentHeading = curr_state[1]

    move_x = MOVEMENTS[previousHeading][0]
    move_y = MOVEMENTS[previousHeading][1]

    a = previousPosition[0] + move_x
    b = previousPosition[1] + move_y

    position = (a, b)

    if movement_policy == 'straight_until_obstacle':
        if position == currentPosition and currentHeading == previousHeading:
            return 0.0
        elif is_obstacle(position):
            if currentPosition == previousPosition:
                return np.log(1/4)
        return -np.inf

    elif movement_policy == 'random_walk':
        if currentPosition == position or currentPosition == previousPosition:
            return np.log(1/4)
        return -np.inf


# ### Viterbi Algorithm
def viterbi(observations, start_state, movement_policy,states,sigma):
    """
    Perform the Viterbi algorithm to find the most likely sequence of states given a series of observations.

    Parameters:
    - observations (list of tuples): A list of observed positions, each as a tuple (obs_x, obs_y).
    - start_state (tuple): The initial state represented as (position, heading),
                           where position is a tuple of (x, y) coordinates.
    - movement_policy (str): The movement policy that dictates how transitions are made.
                             Options are 'straight_until_obstacle' and 'random_walk'.
    - states (list of tuples): A list of all possible states, each represented as (position, heading).
    - sigma (float): The standard deviation of the Gaussian distribution representing observation noise.

    Returns:
    - list of tuples: The most probable sequence of states that could have led to the given observations.
    """
    ###### YOUR CODE HERE ######
    #pass
    memo = {}
    maxi = -np.inf
    result = []

    for state in states:
        p, path = f(observations, len(observations)-1, state, start_state, movement_policy, states, sigma, memo)
        if p > maxi:
            maxi = p
            result = path

    return result


# ### Evaluation (DO NOT CHANGE THIS)
def getestimatedPath(policy, results, states, sigma):
    """
    Estimate the path of the Roomba using the Viterbi algorithm for a specified policy.

    Parameters:
    - policy (str): The movement policy used during simulation, such as 'random_walk' or 'straight_until_obstacle'.
    - results (dict): A dictionary containing simulation results for different policies. Each policy's data includes:
                      - 'true_positions': List of true positions of the Roomba at each time step.
                      - 'headings': List of headings of the Roomba at each time step.
                      - 'observations': List of noisy observations at each time step.
    - states (list of tuples): A list of all possible states (position, heading) for the Hidden Markov Model.
    - sigma (float): The standard deviation of the Gaussian noise used in the emission probability.

    Returns:
    - tuple: A tuple containing:
        1. true_positions (list of tuples): The list of true positions from the simulation.
        2. estimated_path (list of tuples): The most likely sequence of states estimated by the Viterbi algorithm.
    """
    print(f"\nProcessing policy: {policy}")
    data = results[policy]
    observations = data['observations']
    start_state = (data['true_positions'][0], data['headings'][0])
    estimated_path = viterbi(observations, start_state, policy, states, sigma)
    return data['true_positions'], estimated_path


def evaluate_viterbi(estimated_path, true_positions, T,policy):
    """
    Evaluate the accuracy of the Viterbi algorithm's estimated path compared to the true path.
    """
    correct = 0
    for true_pos, est_state in zip(true_positions, estimated_path):
        if true_pos == est_state[0]:
            correct += 1
    accuracy = correct / T * 100
    # data['accuracy'] = accuracy
    print(f"Tracking accuracy for {policy.replace('_', ' ')} policy: {accuracy:.2f}%")


def plot_results(true_positions, observations, estimated_path, policy):
    """
    Plot the true and estimated paths of the Roomba along with the noisy observations.
    The function plots and saves the graphs of the true and estimated paths.
    """
    # Extract coordinates
    true_x = [pos[0] for pos in true_positions]
    true_y = [pos[1] for pos in true_positions]
    obs_x = [obs[0] for obs in observations]
    obs_y = [obs[1] for obs in observations]
    est_x = [state[0][0] for state in estimated_path]
    est_y = [state[0][1] for state in estimated_path]

    # Identify start and end positions
    start_true = true_positions[0]
    end_true = true_positions[-1]
    start_est = estimated_path[0][0]
    end_est = estimated_path[-1][0]

    # Plotting
    plt.figure(figsize=(10, 10))

    # True Path Plot
    plt.subplot(2, 1, 1)
    plt.plot(true_x, true_y, 'g-', label='True Path', linewidth=2)
    plt.scatter(obs_x, obs_y, c='r', s=10, label='Observations')

    # Mark start and end positions on the true path
    plt.scatter(*start_true, c='b', marker='o', s=100, label='True Start', edgecolors='black')
    plt.scatter(*end_true, c='purple', marker='X', s=100, label='True End', edgecolors='black')

    plt.title(f'Roomba Path Tracking ({policy.replace("_", " ").title()} Policy) - True Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)

    # Estimated Path Plot
    plt.subplot(2, 1, 2)
    plt.plot(est_x, est_y, 'b--', label='Estimated Path', linewidth=2)
    plt.scatter(obs_x, obs_y, c='r', s=10, label='Observations')

    # Mark start and end positions on the estimated path
    plt.scatter(*start_est, c='b', marker='o', s=100, label='Estimated Start', edgecolors='black')
    plt.scatter(*end_est, c='purple', marker='X', s=100, label='Estimated End', edgecolors='black')

    plt.title(f'Roomba Path Tracking ({policy.replace("_", " ").title()} Policy) - Estimated Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    
    # fname = f"{policy.replace('_', ' ')}_seed_{seed}_Policy_Roomba_Path_Tracking.png"
    fname = f"{policy.replace('_', ' ')}_Policy_Roomba_Path_Tracking.png"
    plt.savefig(fname)

# Helper function
def f(observations, x, curr, start, movement_policy, states, sigma, memo):

    if x == 0:
        if curr == start:
            p1 = emission_probability(curr, observations[0], sigma)
            return p1, [curr]
        else:
            return -np.inf, []

    if (x, curr) in memo:
        return memo[(x, curr)]

    maxi = -np.inf
    result = []

    for prev in states:
        p1 = transition_probability(prev, curr, movement_policy)
        if p1 == -np.inf:
            continue
        p, prev_path = f(observations, x-1, prev, start, movement_policy, states, sigma, memo)
        p2 = emission_probability(curr, observations[x], sigma)
        p_new = p+p1+p2
        if p_new > maxi:
            maxi = p_new
            result = prev_path+[curr]

    memo[(x, curr)] = (maxi, result)
    return maxi, result


if __name__ == "__main__":
    # # 1. Set up the environment, including grid size, headings, and movements.
    # seeds = [235, 100, 42, 30]
    # results_data = {}
    # for seed in seeds:
    #     print("For seed {}".format(seed))
    #     setup_environment(seed)
    #     sigma = 1.0  # Observation noise standard deviation
    #     T = 50       # Number of time steps
    #
    #     # Simulate for both movement policies
    #     policies = ['random_walk', 'straight_until_obstacle']
    #     results = {}
    #
    #     # 2. Loop through each movement policy and simulate the Roomba's movement:
    #     #    - Generate true positions, headings, and noisy observations.
    #     #    - Store the results in the dictionary.
    #     for policy in policies:
    #         true_positions, headings, observations = simulate_roomba(T, policy,sigma)
    #         results[policy] = {
    #             'true_positions': true_positions,
    #             'headings': headings,
    #             'observations': observations
    #         }
    #
    #     # 3. Define the HMM components
    #     #   - A list (states) containing all possible states of the Roomba, where each state is represented as a tuple ((x, y), h)
    #     #   - x, y: The position on the grid.
    #     #   - h: The heading or direction (e.g., 'N', 'E', 'S', 'W').
    #     states = []
    #     ###### YOUR CODE HERE ######
    #     for x in range(GRID_WIDTH):
    #         for y in range(GRID_HEIGHT):
    #             for h in ['N', 'S', 'E', 'W']:
    #                 states.append(((x, y), h))
    #
    #     # 4. Loop through each policy to estimate the Roomba's path using the Viterbi algorithm:
    #     #    - Retrieve the true positions and estimated path.
    #     #    - Evaluate the accuracy of the Viterbi algorithm.
    #     #    - Plot the true and estimated paths along with the observations.
    #     for policy in policies:
    #         true_positions, estimated_path = getestimatedPath(policy,results,states,sigma)
    #         evaluate_viterbi(estimated_path, true_positions, T,policy)
    #         plot_results(seed, true_positions, results[policy]['observations'], estimated_path, policy)
    #         results_data[(seed, policy)] = estimated_path
    #
    # file_path = "estimated_paths.csv"
    # with open(file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     # Writing the header
    #     writer.writerow(["Seed", "Policy", "Estimated Path"])
    #     # Writing the rows
    #     for (seed, policy), path in results_data.items():
    #         writer.writerow([seed, policy, path])

    # 1. Set up the environment, including grid size, headings, and movements.
    seed = 30
    setup_environment(seed)
    sigma = 1.0  # Observation noise standard deviation
    T = 50  # Number of time steps

    # Simulate for both movement policies
    policies = ['random_walk', 'straight_until_obstacle']
    results = {}

    # 2. Loop through each movement policy and simulate the Roomba's movement:
    #    - Generate true positions, headings, and noisy observations.
    #    - Store the results in the dictionary.
    for policy in policies:
        true_positions, headings, observations = simulate_roomba(T, policy, sigma)
        results[policy] = {
            'true_positions': true_positions,
            'headings': headings,
            'observations': observations
        }

    # 3. Define the HMM components
    #   - A list (states) containing all possible states of the Roomba, where each state is represented as a tuple ((x, y), h)
    #   - x, y: The position on the grid.
    #   - h: The heading or direction (e.g., 'N', 'E', 'S', 'W').
    states = []
    ###### YOUR CODE HERE ######
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            for h in ['N', 'S', 'E', 'W']:
                states.append(((x, y), h))
    # 4. Loop through each policy to estimate the Roomba's path using the Viterbi algorithm:
    #    - Retrieve the true positions and estimated path.
    #    - Evaluate the accuracy of the Viterbi algorithm.
    #    - Plot the true and estimated paths along with the observations.
    for policy in policies:
        true_positions, estimated_path = getestimatedPath(policy, results, states, sigma)
        evaluate_viterbi(estimated_path, true_positions, T, policy)
        plot_results(true_positions, results[policy]['observations'], estimated_path, policy)
