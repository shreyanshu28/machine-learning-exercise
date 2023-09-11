import os
import torch
from collections import deque
import numpy as np

# from .rule_action import

from .train import Agent, state_to_features, ACTIONS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def setup(self):
    # stuff for rule based actions
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    self.walkedTiles = []

    continueTraining = False
    model_path = 'tmp/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print("Created model path.")

    if (not continueTraining and self.train): # or not os.path.isfile(model_path)
        self.logger.info("Setting up model from scratch.")
        print("Setting up model from scratch.")
        self.agent = Agent(gamma=0.1, epsilon=1., lr=1e-7,
                              input_dims=(17, 17, 1), eps_dec=1e-5,
                              n_actions=6, max_mem_size=100000, batch_size=64,
                              eps_end=0.1, replace=2000, fname=model_path, train=self.train, use_cpu=False)
    elif continueTraining and self.train:
        self.agent = Agent(gamma=0.1, epsilon=1., lr=1e-7,
                              input_dims=(17, 17, 1), eps_dec=1e-5,
                              n_actions=6, max_mem_size=100000, batch_size=64,
                              eps_end=0.1, replace=2000, fname=model_path, train=self.train, use_cpu=False)
        self.agent.load_model()

    else:
        self.logger.info("Loading model from saved state.")
        print("Loading model from saved state.")
        self.agent = Agent(gamma=0.1, epsilon=1., lr=1e-6,
                           input_dims=(17, 17, 1), eps_dec=1e-5,
                           n_actions=6, max_mem_size=100000, batch_size=64,
                           eps_end=0.1, replace=2000, fname=model_path, train=self.train, use_cpu=True)
        self.agent.load_model()


def act(self, game_state: dict) -> str:
    # observation = state_to_features(game_state)
    actions = ACTIONS[self.agent.choose_action(game_state, self)]
    # print(actions)
    return actions
