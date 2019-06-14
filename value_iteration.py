import pandas as pd
import numpy as np

class MDP():
    def __init__(self, states, actions, transitions, rewards, discount):
        """
        Parameters:
            states - ndarray of state names
            actions - ndarray of action names
            transitions - ndarray of [action, state]
        """
        self.S = states
        self.A = actions
        self.transitions = transitions
        self.R = rewards
        self.discount = discount

def value_iteration(mdp, age, epsilon):
    """
    Parameters:
        mdp - an MDP object with states S, actions A(s), transition model P(s'|s,a), rewards R(s), discount gamma

    Returns:
        U(s) - a map from each state in S to a utility value
    """

    # Initial utility map sends all states to 0
    U = {s:0 for s in mdp.S}
    delta = np.inf
    U_next = U.copy()
    while(True):
        U = U_next.copy()
        delta = 0
        for state in mdp.S:
            U_next[state] = mdp.R(state) + mdp.discount * get_max_action_util(state,age,mdp,U)

            if abs(U_next[state] - U[state]) > delta:
                delta = abs(U_next[state] - U[state])

        if delta < epsilon:
            break

        print('delta:',delta)

    return U

def get_max_action_util(state,age_, mdp,U):

    u_per_action = np.zeros(len(mdp.A))
    for i, a_ in enumerate(mdp.A):
        total = 0
        for s in mdp.S:
            p = float(mdp.transitions.query("health == @state & action == @a_ & age == @age_ ")[s])
            u = U[s]
            total += u * p
        u_per_action[i] = total
    return max(u_per_action)

def get_best_action(state, age_,mdp,U):
    u_per_action = np.zeros(len(mdp.A))
    for i, a_ in enumerate(mdp.A):
        total = 0
        for s in mdp.S:
            p = float(mdp.transitions.query("health == @state & action == @a_ & age == @age_ ")[s])
            u = U[s]
            total += u * p
        u_per_action[i] = total
    return mdp.A[np.argmax(u_per_action)]


def prob1(age = 30, discount = .9):
    data = pd.read_csv('transition1.csv')
    # Define states, actions by slicing 'health', 'actions'
    states = data.health.unique()
    actions = data.action.unique()
    def reward(state):
        if state == "Dead":
            return -100
        if state == 'noAAA':
            return 100 - age
        return .9 * (100 - age)

    # Build the transition matrix using slices of the data
    mdp = MDP(states, actions, data, reward, discount)
    utilities = value_iteration(mdp, age, 1)
    print('for age:', age)
    print('for discount:', discount)
    return {u:a for u,a in zip(utilities.keys(), [get_best_action(s,age,mdp,utilities) for s in mdp.S]) }

def prob2():
    data = pd.read_csv('transition2.csv')
