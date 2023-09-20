import os
import pickle
import random
import hashlib

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    continue_training = False
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.Q = {}
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.Q = pickle.load(file)
    print("Setup done.")

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    #print(hashlib.sha256(state_to_features(game_state)).hexdigest())
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        #use loaded model

        # return action

    observation = state_to_features(game_state)
    if observation not in self.Q:
        return np.random.choice(ACTIONS,p=[0.25,0.25,0.25,0.25,0,0])
    
    self.logger.debug("Querying model for action.")
    return ACTIONS[np.argmax(self.Q[observation])]


def state_to_features(state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if state is None:
        return None

    
    #def state_to_features(state: dict) -> np.array:
    cols, rows = state['field'].shape[0], state['field'].shape[1]
    observation = np.zeros([rows, cols], dtype=np.float32)

    observation[:, :] = state['field']*10

    if state['coins']:
        coins_x, coins_y = zip(*state['coins'])
        observation[list(coins_y), list(coins_x)] = 20  # revealed coins

    if state['bombs']:
        bombs_xy, bombs_t = zip(*state['bombs'])
        bombs_x, bombs_y = zip(*bombs_xy)
        observation[list(bombs_y), list(bombs_x)] = [bt+1 for bt in  list(bombs_t)]

    if state['self']:  # let's hope there is...
        _, _, _, (self_x, self_y) = state['self']
        observation[self_y, self_x] = 30

    if state['others']:
        _, _, others_bomb, others_xy = zip(*state['others'])
        others_x, others_y = zip(*others_xy)
        observation[others_y, others_x] = -40
        # missing if others can place bomb

    observation[np.where(state['explosion_map'] != 0)[1],np.where(state['explosion_map'] != 0)[0]] = -2
    #observation += np.where(state['explosion_map'], state['explosion_map'] * -2, state['explosion_map']).reshape(rows, cols)
    #print(state['explosion_map'])
    #print(np.where(state['explosion_map'] != 0)[0])
    #print(np.array(observation))
    return hashlib.sha256(observation).hexdigest()