from __future__ import print_function
import numpy as np
from F9utils import F9GameClient
from F9utils import RLAgent


class SimpleAgent(RLAgent):
    def __init__(self,
                 client,
                 state = None,
                 learning_rate=0.15,
                 discount_factor=0.9,
                 exploration_rate=0.6,
                 exploration_decay_rate=0.96):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.state = state
        self._num_actions = 8
        self.action = np.random.randint(0, self._num_actions - 1)

        self.client = client

        self.__boundaries = [
           # (0, 999), #fuel
            (-50, 50), # vx
            (0, 60), #dist
            (-0.3, 0.3), #angle
            (0, 100), # px,
            (-50, 1) #vy
        ]

        self.num_discrete_states = 8
        self._discrete_states = [np.linspace(low, up, self.num_discrete_states) for (low, up) in self.__boundaries]
        self._len_discrete_states = self.num_discrete_states ** len(self._discrete_states)
        self.q = np.zeros((self._len_discrete_states, self._num_actions))


    def getAction(self, state, reward):
        next_state = self._build_state(state)

        enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)

        next_action = np.random.randint(0, self._num_actions - 1)

        if not enable_exploration:
            next_action = np.argmax(self.q[next_state])

        return next_action

    def provideFeedback(self, state, action, reward, new_state):
        state = self._build_state(state)
        new_state = self._build_state(new_state)
        self.q[state, action] = (1 - self.learning_rate) * self.q[state, action] \
                                          + (self.learning_rate * (reward + self.discount_factor
                                                                   * self.q[new_state, np.argmax(self.q[new_state])]))

    def _build_state(self, observation):

        observation = [observation[0]['vx'], observation[0]['dist'], observation[0]['angle'], observation[0]['px'],
                       observation[0]['vy']]

        states = [np.digitize(val, self._discrete_states[i]) * (len(self._discrete_states) ** i) for i, val in
                  enumerate(observation)]
        return sum(states)


def solve():
    # Setup agent
    client = F9GameClient()
    state = client.curState
    ai = SimpleAgent(client, state=state)
     # Observe current state
    reward = 0
    while True:
        action = ai.getAction(state, reward)
        actions = client.actions(state)
        act_arr = actions[action]# Decide what to do
        client.doAction(act_arr)                                 # Act
        new_state = client.curState                             # Observe new state
        reward = client.getReward(new_state)                    # Observe reward
        ai.provideFeedback(state, action, reward, new_state)    # Provide feeback to the agent

        agent, platform, system = new_state
        print("Agent state %s\n Platform state %s\n System state %s\n Reward %s\n" % (agent, platform, system, reward))

        if client.isTerminalState(new_state):
            client.reset_game()
            state = client.curState
        else:
            state = new_state

if __name__ == "__main__":
    solve()

# -------------------------------------------------- #
# --------------- you have landed ------------------ #
# -------------------------------------------------- #
