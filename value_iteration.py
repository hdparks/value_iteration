
class MDP():
    def __init__(self, states,  actions, transitions, rewards, discount):
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
    U = {s:0 for s in mdp.S}
