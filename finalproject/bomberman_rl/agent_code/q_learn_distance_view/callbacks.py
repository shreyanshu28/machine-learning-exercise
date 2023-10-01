import os
import pickle
import json
import random
import hashlib

import numpy as np

from .import perspective  as p


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
    self.file_path = f"save_files/my-saved-model_distance_view.json"
    if not os.path.isfile(self.file_path):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.Q = {}
    else:
        self.logger.info("Loading model from saved state.")
        print(f"loaded: {self.file_path}")
        with open(self.file_path, "rb") as file:
            self.Q = json.load(file)
        print(list(self.Q.items())[:5])
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
    random_prob = .3
    #print(hashlib.sha256(state_to_features(game_state)).hexdigest())
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, 0, .2])
        #use loaded model

        # return action

    observation = p.state_to_features(game_state)
    if observation not in self.Q:
        return np.random.choice(ACTIONS,p=[.2, .2, .2, .2, .1, .1])
    
    self.logger.debug("Querying model for action.")
    return ACTIONS[np.argmax(self.Q[observation])]


