import os
import pickle
import json
import random
import hashlib

import numpy as np
from . import helper as h

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# bomb info


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
    self.limit_to = 3
    self.epsilon = 0.00001
    #file_path = f"save_files/my-saved-model_view{self.limit_to}_test.json"
    self.file_path = f"save_files/my-coin-model_view{self.limit_to}.pt"
    if not os.path.isfile(self.file_path):
        #self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.Q = {}
    else:
        #self.logger.info("Loading model from saved state.")
        print(f"loaded: {self.file_path}")
        with open(self.file_path, "rb") as file:
            self.Q = pickle.load(file)
        print(*list(self.Q.items())[:10], sep="\n")
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

    #print(hashlib.sha256(state_to_features(game_state)).hexdigest())
    if self.train and random.random() < self.epsilon:
        #self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .15, .05])
        #use loaded model

        # return action

    observation = state_to_features(self,game_state)
    if observation not in self.Q:
        #self.logger.info(f"random for{observation}")
        return np.random.choice(ACTIONS,p=[.2, .2, .2, .2, .1, .1])
    
    #self.logger.info(f"{self.Q[observation]}")
    #self.logger.debug("Querying model for action.")
    return ACTIONS[np.argmax(self.Q[observation])]


def state_to_features(self,state: dict) -> np.array:
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
    rows, cols = state['field'].shape[0], state['field'].shape[1]
    playerLoc = state['self'][3]


    observation = np.zeros([rows, cols], dtype=np.float32)


    if state['coins']:
        coins_x, coins_y = zip(*state['coins'])
        dist_to_coin = h.calc_dir(state['coins'], playerLoc)
        observation[list(coins_x), list(coins_y)] = 20  # revealed coins

    if state['self']:  # let's hope there is...
            _, _, self_bomb, (self_x, self_y) = state['self']
            observation[self_x, self_y] = 40 if self_bomb else 30
    
    if state['bombs']:
        for bomb in state['bombs']:
            (x,y), bt = bomb
            blast_coord = h.get_blast_coords(x,y,state['field'])
            #print(blast_coord.shape)
            for coord in blast_coord:
                observation[coord] = bt+1
                if coord == (self_x,self_y):
                    observation[coord] = 50
        #dist_to_bomb = h.calc_dir(bombs_xy,playerLoc)

    if state['others']:
        _, _, others_bomb, others_xy = zip(*state['others'])
        #print('others')
        dist_to_enemy = h.calc_dir(others_xy,playerLoc)
        others_x, others_y = zip(*others_xy)
        for i in range(len(others_bomb)):
            observation[others_x[i], others_y[i]] = -40 if others_bomb[i] else -30

    observation[np.where(state['explosion_map'] != 0)[0],np.where(state['explosion_map'] != 0)[1]] = -2

    observation[state['field'] != 0] = state['field'][state['field'] != 0]*10

    if state['self']:  # reduce view
        _, _, _, (self_x, self_y) = state['self']
        observation = observation[max(1,self_x - self.limit_to):min(self_x + self.limit_to,rows-1),\
                                   max(1,self_y - self.limit_to):min(self_y + self.limit_to,cols-1)]


    #observation += np.where(state['explosion_map'], state['explosion_map'] * -2, state['explosion_map']).reshape(rows, cols)
    #print(state['explosion_map'])
    #print(np.where(state['explosion_map'] != 0)[0])
    return tuple(map(tuple,observation))
