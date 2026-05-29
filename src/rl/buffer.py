class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, s, a, r, d, lp, v):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.dones.append(d)
        self.log_probs.append(lp)
        self.values.append(v)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)
