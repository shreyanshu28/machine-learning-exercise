'''
To play against random agents: 
python main.py play --agents sac random_agent random_agent random_agent --skip-frames

To play against rule based agents:
python main.py play --agents sac rule_based_agent rule_based_agent rule_based_agent --skip-frames

To train the agent for n-rounds against rule based agents:
python main.py play --my-agent sac --train 1 --n-rounds 1000 --no-gui 

'''
from collections import namedtuple, deque

import pickle
from typing import List
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import events as e
import settings as s
from .rule_action import rule_action
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRAIN_EVERY = 1
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
PLACEHOLDER_EVENT = "PLACEHOLDER"
REDUCED_DISTANCE_TO_NEXT_COIN_EVENT = "REDUCED_DISTANCE_TO_COIN"
INCREASED_DISTANCE_TO_NEXT_COIN_EVENT = "INCREASED_DISTANCE_TO_COIN"
REDUCED_DISTANCE_TO_ENEMY_EVENT = "REDUCED_DISTANCE_TO_ENEMY"
INCREASED_DISTANCE_TO_BOMB_EVENT = "INCREASED_DISTANCE_TO_BOMB"
NEW_TILE_FOUND = "DISCOVERED_NEW_TILE"





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
    self.train_every = 1
    self.save_every = 10
    self.warmup = 100
    s.MAX_STEPS = 300



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

    if new_game_state['self'][3] not in self.walkedTiles:
        self.walkedTiles.append(new_game_state['self'][3])
        events.append(NEW_TILE_FOUND)

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
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 500,
        e.BOMB_DROPPED: 10,
        e.COIN_FOUND: 5,
        e.SURVIVED_ROUND: 50,
        e.CRATE_DESTROYED: 5,
        e.MOVED_LEFT: 2,
        e.MOVED_RIGHT: 2,
        e.MOVED_UP: 2,
        e.MOVED_DOWN: 2,
        e.INVALID_ACTION: -40,
        e.WAITED: 0,
        e.GOT_KILLED: -100,
        e.KILLED_SELF: -200,
        NEW_TILE_FOUND: 10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.walkedTiles = []
    self.agent.store_transition(state_to_features(last_game_state), ACTIONS.index(last_action),
                                reward_from_events(self, events), state_to_features(last_game_state), True)


    if last_game_state['round'] % (self.save_every) == 0:
        self.agent.save_model()

    #follow the training here based on the events
    if self.train:
        actor = Actor().to(device)
        qf1 = SoftQNetwork().to(device)
        qf2 = SoftQNetwork().to(device)
        qf1_target = SoftQNetwork().to(device)
        qf2_target = SoftQNetwork().to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)
        #extract data and import the model (figure out how to do this)
        data = ...
        # CRITIC training
        with torch.no_grad():
            _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
            qf1_next_target = qf1_target(data.next_observations)
            qf2_next_target = qf2_target(data.next_observations)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

        # use Q-values only for the taken actions
        qf1_values = qf1(data.observations)
        qf2_values = qf2(data.observations)
        qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
        qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()


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

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x / 255.0))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs

