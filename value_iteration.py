import pandas as pd

class MDP():
    def __init__(self, rewards, discount):
        """
        Parameters:
            states - ndarray of state names
            actions - ndarray of action names
            transitions - ndarray of [action, state]
        """
        self.S = states
        self.A = actions
        self.transitions = transitions
        self.rewards = rewards
        self.discount = discount

def value_iteration(mdp, epsilon):
    """
    Parameters:
        mdp - an MDP object with states S, actions A(s), transition model P(s'|s,a), rewards R(s), discount gamma

    Returns:
        U(s) - a map from each state in S to a utility value
    """

    # Initial utility map sends all states to 0
    U = np.zeros(len(self.S))
    delta = 0
    while(True):
        for state in mdp.S:
            U_next[state] = mdp.R(state) + mdp.discount * max([sum([mdp.transitions[action,state,s_next] * U[s_next] for s_next in mdp.S]) for action in mdp.A )

            if abs(U_next[state] - U[state]) > delta:
                delta = abs(U_next[state] - U[state])

        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break

    return U
