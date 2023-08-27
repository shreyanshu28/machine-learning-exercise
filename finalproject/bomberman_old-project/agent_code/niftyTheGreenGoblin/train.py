from collections import namedtuple, deque

import pickle
from typing import List
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


import numpy as np
import events as e
import settings as s
from .rule_action import rule_action

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

LSTM_MEMORY = 64

# Hyper parameters -- DO modify
TRAIN_EVERY = 10
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
PLACEHOLDER_EVENT = "PLACEHOLDER"
REDUCED_DISTANCE_TO_NEXT_COIN_EVENT = "REDUCED_DISTANCE_TO_COIN"
INCREASED_DISTANCE_TO_NEXT_COIN_EVENT = "INCREASED_DISTANCE_TO_COIN"
REDUCED_DISTANCE_TO_ENEMY_EVENT = "REDUCED_DISTANCE_TO_ENEMY"
INCREASED_DISTANCE_TO_BOMB_EVENT = "INCREASED_DISTANCE_TO_BOMB"





'''
gamma=0.99, epsilon=1.0, lr=1e-3,                                            input_dims=(17, 17, 1), epsilon_dec=1e-6,                                            n_actions=6, mem_size=100000, batch_size=64,                                            epsilon_end=0.01, fname='dqn_model_bombalistic.h5'
'''
# fill data

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']


def setup_training(self):
    """
     Initialise self for training purpose.

     This is called after `setup` in callbacks.py.

     :param self: This object is passed to all callbacks and you can set arbitrary values.
     """
    self.train_every = 10
    self.save_every = 500
    self.warmup = 0
    s.MAX_STEPS = 200



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is None:
        events.append(PLACEHOLDER_EVENT)
        old_game_state = new_game_state

    if any((e.GOT_KILLED, e.KILLED_SELF, e.SURVIVED_ROUND)) in events:
        done = True
    else:
        done = False

    # reward = reward_from_events(self, events)


    self.agent.store_transition(state_to_features(old_game_state), ACTIONS.index(self_action),
                                reward_from_events(self, events), state_to_features(new_game_state), done)

    if self.warmup >= old_game_state['round']:
        return
    if self.train:
        self.agent.learn()


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        # e.COIN_COLLECTED: 1,
        # e.KILLED_OPPONENT: 5,
        # e.BOMB_DROPPED: 0.001,
        # e.COIN_FOUND: 0.01,
        # e.SURVIVED_ROUND: 0.5,
        # e.CRATE_DESTROYED: 0.4,
        # e.MOVED_LEFT: 0.01,
        # e.MOVED_RIGHT: 0.01,
        # e.MOVED_UP: 0.01,
        # e.MOVED_DOWN: 0.01,
        # e.INVALID_ACTION: -2,
        # e.WAITED: -0.02,
        # e.GOT_KILLED: -1,
        # e.KILLED_SELF: -5
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.BOMB_DROPPED: -0.004,
        e.COIN_FOUND: 0.01,
        e.SURVIVED_ROUND: 0.5,
        e.CRATE_DESTROYED: 0.1,
        e.MOVED_LEFT: 0.001,
        e.MOVED_RIGHT: 0.001,
        e.MOVED_UP: 0.001,
        e.MOVED_DOWN: 0.001,
        e.INVALID_ACTION: -0.02,
        e.WAITED: -0.005,
        e.GOT_KILLED: -1,
        e.KILLED_SELF: -1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.agent.store_transition(state_to_features(last_game_state), ACTIONS.index(last_action),
                                reward_from_events(self, events), state_to_features(last_game_state), True)


    if last_game_state['round'] % (self.save_every) == 0:
        self.agent.save_model()


def state_to_features(state: dict) -> np.array:
    cols, rows = state['field'].shape[0], state['field'].shape[1]
    observation = np.zeros([rows, cols, 1], dtype=np.float32)

    observation[:, :, 0] = state['field']

    if state['coins']:
        coins_x, coins_y = zip(*state['coins'])
        observation[list(coins_y), list(coins_x), 0] = 2  # revealed coins

    if state['bombs']:
        bombs_xy, bombs_t = zip(*state['bombs'])
        bombs_x, bombs_y = zip(*bombs_xy)
        observation[list(bombs_y), list(bombs_x), 0] = -2  # list(bombs_t)

    if state['self']:  # let's hope there is...
        _, _, _, (self_x, self_y) = state['self']
        observation[self_y, self_x, 0] = 3

    if state['others']:
        _, _, _, others_xy = zip(*state['others'])
        others_x, others_y = zip(*others_xy)
        observation[others_y, others_x, 0] = -3

    observation += np.where(state['explosion_map'], state['explosion_map'] * -4, state['explosion_map']).reshape(rows, cols, 1)
    # print(state['explosion_map'])
    return observation


class DQN(nn.Module):
    def __init__(self, lr, input_dims, fc2_dims, n_actions, name, chkpt_dr, use_cpu=False):
        super(DQN, self).__init__()
        self.checkpoint_dr = chkpt_dr
        self.chkptf = os.path.join(self.checkpoint_dr, name)
        c = input_dims[0]
        self.input_dims = input_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Conv2d(c, c, kernel_size=(3, 3), padding='same')
        self.fc2 = nn.Conv2d(c, c, kernel_size=(3, 3), padding='same')
        self.fc3 = nn.Linear(c**2, self.fc2_dims)
        self.fc4 = nn.Linear(self.fc2_dims, 2*self.fc2_dims)
        self.fc5 = nn.Linear(2*self.fc2_dims, 2*self.fc2_dims)
        self.fc6 = nn.Linear(2*self.fc2_dims, self.fc2_dims)
        self.rnn = nn.LSTMCell(self.fc2_dims, LSTM_MEMORY)
        self.after_rnn = nn.Linear(LSTM_MEMORY, 128)
        self.fc7 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()
        self.dropout = nn.Dropout(0.3)
        if not use_cpu:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.to(self.device)

    def forward(self, state, hn, cn):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        # x = self.dropout(x)
        x = F.relu(self.fc6(x))
        hn, cn = self.rnn(x, (hn, cn))
        x = hn
        x = F.relu(self.after_rnn(x))
        actions = self.fc7(x)
        return actions, hn.detach(), cn.detach()

    def save_chkpt(self, final=False):
        # if final:
        #     self.zero_grad(set_to_none=True)
        torch.save(self.state_dict(), self.chkptf)

    def load_chkpt(self):
        self.load_state_dict(torch.load(self.chkptf, map_location=self.device))

    # def init_states(self) -> [Variable, Variable]:
    #     hidden_state = Variable(torch.zeros(1, 64, LSTM_MEMORY).to(self.device))
    #     cell_state = Variable(torch.zeros(1, 64, LSTM_MEMORY).to(self.device))
    #     return hidden_state, cell_state
    #
    # def reset_states(self, hidden_state, cell_state):
    #     hidden_state[:, :, :] = 0
    #     cell_state[:, :, :] = 0
    #     return hidden_state, cell_state


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, fname,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, replace=1000, train=False, use_cpu=False):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.learn_step_cntr = 0
        self.replace = replace
        self.model_file = fname
        self.use_cpu = use_cpu
        self.train = train

        self.Q_eval = DQN(self.lr, n_actions=n_actions, input_dims=input_dims, fc2_dims=256,
                          name='niftyTheGreenGoblin_online1.pt', chkpt_dr=self.model_file, use_cpu=self.use_cpu)

        self.Q_target = DQN(self.lr, n_actions=n_actions, input_dims=input_dims, fc2_dims=256,
                            name='niftyTheGreenGoblin_offline1.pt', chkpt_dr=self.model_file, use_cpu=self.use_cpu)

        self.hns, self.cns = [torch.zeros(1, LSTM_MEMORY)], [torch.zeros(1, LSTM_MEMORY)]

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int64)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)

        self.mem_cntr += 1

    def choose_action(self, game_state, this):
        observation = state_to_features(game_state)
        if self.train:
            if np.random.random() > self.epsilon:
                state = torch.tensor([observation]).to(self.Q_eval.device)
                actions = self.Q_eval.forward(state, self.cns, self.hns)[0]
                action = torch.argmax(actions).item()
                # print('deep action')
            else:
                # action = np.random.choice(self.action_space)
                a = rule_action(this, game_state)
                if a is None:
                    a = 'WAIT'
                action = ACTIONS.index(a)
                # print('random action')
        else:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state, self.cns, self.hns)[0]
            action = torch.argmax(actions).item()
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        self.replace_target_network()



        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]



        q_eval, cn, hn = self.Q_eval.forward(state_batch, self.cns[batch], self.hns[batch])
        self.cns.append(torch.tensor(hn))
        self.cns.append(torch.tensor(cn))
        q_eval = q_eval[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch, self.cns[batch], self.hns[batch])
        q_next[0][terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next[0], dim=1)[0].detach()


        loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.learn_step_cntr += 1

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end

    def replace_target_network(self):
        if self.replace is not None and self.learn_step_cntr % self.replace == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def save_model(self, final=False):

        self.Q_eval.save_chkpt(final=final)
        self.Q_target.save_chkpt(final=final)

    def load_model(self):

        self.Q_eval.load_chkpt()
        self.Q_target.load_chkpt()
