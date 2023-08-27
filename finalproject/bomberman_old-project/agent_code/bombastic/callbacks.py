import os
import torch
import numpy as np

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

# das ist eine Memberfunction des Agent, die muss in der Klasse beleiben
# def load_model(self):


def setup(self):
    self.device = 'cpu'
    self.model_file = "dqn_bombastic.pt"
    self.logger.info("Loading model from saved state.")
    print("Loading model from saved state.")
    load_model(self)


def act(self, game_state: dict) -> str:
    # if np.random.random() < self.agent.epsilon:
    # 	return np.random.choice(ACTIONS)
    # # maybe?
    # # bitte so umschreiben, dass das nur nen game state braucht
    # actions = self.q_net(state_to_features(game_state)[np.newaxis, :])
    # action = np.argmax(actions)
    # return str(action)
    observation = state_to_features(game_state)
    # print(self.agent.choose_action(observation))
    actions = self.q_net(observation[np.newaxis, :])
    action = np.argmax(actions.detach().cpu().numpy())
    return ACTIONS[action]

def load_model(self):
    self.q_net = torch.load_state_dict(self.model_file)
    self.q_net = self.q_net.to(self.device)


def state_to_features(state: dict) -> np.array:
    """
		Build a tensor of the observed board state for the agent.
		Layers:
		0: field with walls and crates
		1: revealed coins
		2: bombs
		3: agents (self and others)
		Returns: observation tensor
		"""
    cols, rows = state['field'].shape[0], state['field'].shape[1]
    observation = np.zeros([rows, cols, 1], dtype=np.float32)

    # write field with crates
    observation[:, :, 0] = state['field']

    # write revealed coins
    if state['coins']:
        coins_x, coins_y = zip(*state['coins'])
        observation[list(coins_y), list(coins_x), 0] = 2  # revealed coins

    # write ticking bombs
    if state['bombs']:
        bombs_xy, bombs_t = zip(*state['bombs'])
        bombs_x, bombs_y = zip(*bombs_xy)
        observation[list(bombs_y), list(bombs_x), 0] = -2  # list(bombs_t)

    """
		bombs_xy = [xy for (xy, t) in state['bombs']]
		bombs_t = [t for (xy, t) in state['bombs']]
		bombs_x, bombs_y = [x for x, y in bombs_xy], [y for x, y in bombs_xy]
		observation[2, bombs_x, bombs_y] = bombs_t or 0
		"""

    # write agents
    if state['self']:  # let's hope there is...
        _, _, _, (self_x, self_y) = state['self']
        observation[self_y, self_x, 0] = 3

    if state['others']:
        _, _, _, others_xy = zip(*state['others'])
        others_x, others_y = zip(*others_xy)
        observation[others_y, others_x, 0] = -3

    # print(observation)
    return torch.from_numpy(observation)
