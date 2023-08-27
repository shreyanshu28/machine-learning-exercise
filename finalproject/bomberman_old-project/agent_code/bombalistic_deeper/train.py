import os
import pickle
from collections import namedtuple, deque
from random import shuffle

from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import events as e


ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRAIN_EVERY = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
PLACEHOLDER_EVENT = "PLACEHOLDER"
REDUCED_DISTANCE_TO_NEXT_COIN = "REDUCED_DISTANCE_TO_COIN"
INCREASED_DISTANCE_TO_NEXT_COIN = "INCREASED_DISTANCE_TO_COIN"
REDUCED_DISTANCE_TO_ENEMY = "REDUCED_DISTANCE_TO_ENEMY"
INCREASED_DISTANCE_TO_BOMB = "INCREASED_DISTANCE_TO_BOMB"

ENEMY_SCORE_INCREASED= "ENEMY_SCORE_INCREASED"

# repeated actions
REPEATED_ACTION = "REPEATED ACTION"
SURVIVED_BOMB = "SURVIVED_BOMB"

# class WorldWrapper:
# class to wrap game_state


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        # positive auxiliary rewards
        e.BOMB_DROPPED: 0.01,
        e.COIN_FOUND: 0.01,
        # e.SURVIVED_ROUND: 0.5,
        e.CRATE_DESTROYED: 0.01,
        e.MOVED_LEFT: 0.001,
        e.MOVED_RIGHT: 0.001,
        e.MOVED_UP: 0.001,
        e.MOVED_DOWN: 0.001,
        # negative auxiliary rewards
        e.INVALID_ACTION: -200,
        e.WAITED: -0.1,
        e.GOT_KILLED: -1,
        e.KILLED_SELF: -0.1,
        e.SURVIVED_ROUND: 3,
        SURVIVED_BOMB: 1,
        '''
        #INCREASED_DISTANCE_TO_BOMB: .2,
        REDUCED_DISTANCE_TO_ENEMY: .02,
        REDUCED_DISTANCE_TO_NEXT_COIN: .002
        #INCREASED_DISTANCE_TO_NEXT_COIN: -.0001,
        #SURVIVED_BOMB: .01,
        #ENEMY_SCORE_INCREASED: -1
        ''': 15
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def setup_training(self):
    """
     Initialise self for training purpose.

     This is called after `setup` in callbacks.py.

     :param self: This object is passed to all callbacks and you can set arbitrary values.
     """
    #self.train_every = 25
    self.save_every = 1000
    self.warmup = 10000

    # DEVICE = self.device



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is None:
        events.append(PLACEHOLDER_EVENT)
        old_game_state = new_game_state
    if (len(old_game_state['bombs']) > len(new_game_state['bombs']) and new_game_state['self']):
        #print(SURVIVED_BOMB)
        events.append(SURVIVED_BOMB)

    def euclid(list, tuple):
        return [np.sqrt((list[i][0] - tuple[0]) ** 2 + (list[i][1] - tuple[1]) ** 2) for i in range(len(list))]

    if (self_action in ['UP', 'DOWN', 'LEFT', 'RIGHT']):
        oldPlayerLoc, newPlayerLoc = old_game_state['self'][3], new_game_state['self'][3]
        if len(old_game_state['coins']) != 0 and len(new_game_state['coins']) != 0:
            if min(euclid(old_game_state['coins'], oldPlayerLoc)) >= min(euclid(new_game_state['coins'], newPlayerLoc)):
                events.append(REDUCED_DISTANCE_TO_NEXT_COIN)
            else:
                events.append(INCREASED_DISTANCE_TO_NEXT_COIN)

        if len(old_game_state['others']) != 0:
            others = [xy for (n, s, b, xy) in old_game_state['others']]
            if min(euclid(others, oldPlayerLoc)) >= min(euclid(others, newPlayerLoc)):
                events.append(REDUCED_DISTANCE_TO_ENEMY)


    if any((e.GOT_KILLED, e.KILLED_SELF, e.SURVIVED_ROUND)) in events:
        done = True
    else:
        done = False
    # print(reward_from_events(self, events))
    self.agent.store_transition(state_to_features(old_game_state), ACTIONS.index(self_action),
                                reward_from_events(self, events), state_to_features(new_game_state), done)

    # if old_game_state['round'] % TRAIN_EVERY == 0:
    #     self.agent.learn()
    #     self.agent.save_model()

    #  self.agent.store_transition(state_to_features(old_game_state), ACTIONS.index(self_action),
    #                            reward_from_events(self, events), state_to_features(new_game_state), False)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    #print("\n", last_game_state['self'][1])
    # print('last game state: ', last_game_state)
    self.agent.store_transition(state_to_features(last_game_state), ACTIONS.index(last_action),
                                reward_from_events(self, events), state_to_features(last_game_state), True)

    # if self.train and last_game_state['round'] % self.train_every == 0:
    # self.agent.learn()
    if last_game_state['round'] == self.warmup:# and not os.path.isfile('observation.npy'):
        with open("observation.pt", "wb") as file:
            obs = [self.agent.state_memory, self.agent.new_state_memory, self.agent.action_memory,
                   self.agent.reward_memory,
                   self.agent.terminal_memory]
            pickle.dump(obs, file)
        #np.save('observation',self.agent.)
    if last_game_state['round'] % (self.save_every) == 0:
        self.agent.save_model()


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

    observation += np.where(state['explosion_map'], state['explosion_map'] * -4, state['explosion_map']).reshape(rows, cols, 1)
    # print(np.where(state['explosion_map'], state['explosion_map'] * -4, state['explosion_map']).reshape(rows, cols, 1))

    # print(observation)
    return observation


class DQN(nn.Module):

    def __init__(self, lr, input_dims,  n_actions, name, fc2_dims = 64 ,fc3_dims = 32):
        super(DQN, self).__init__()
        self.chkptf = name
        c = input_dims[0]
        self.input_dims = input_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Conv2d(c, c, kernel_size=(3, 3), padding='same')
        self.fc2 = nn.Conv2d(c, c, kernel_size=(3, 3), padding='same')
        self.fc3 = nn.Linear(c ** 2, self.fc2_dims)
        self.fc4 = nn.Linear(self.fc2_dims, 2 * self.fc2_dims)
        self.fc5 = nn.Linear(2 * self.fc2_dims, 2 * self.fc2_dims)
        self.fc6 = nn.Linear(2 * self.fc2_dims, self.fc2_dims)
        self.fc7 = nn.Linear(self.fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        # self.loss = nn.SmoothL1Loss()
        self.dropout = nn.Dropout(0.3)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        actions = self.fc7(x)
        return actions
    '''
        self.chkptf = name
        c = input_dims[0]
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1 = nn.Conv2d(c, c, kernel_size=(3, 3), padding='same')
        self.fc2 = nn.Linear(c**2, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.fc4 = nn.Linear(fc3_dims, 32)
        self.fc5 = nn.Linear(32, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        actions = self.fc5(x)
        return actions
    '''
    def save_chkpt(self):
        torch.save(self, self.chkptf)

    def load_chkpt(self):
        self = torch.load(self.chkptf)


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, fname,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, replace=1000, train=False):
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
        self.train = train

        self.Q_eval = DQN(self.lr, n_actions=n_actions, input_dims=input_dims,
                          name='bombalistic_eval.pt')

        self.Q_target = DQN(self.lr, n_actions=n_actions, input_dims=input_dims,
                            name='bombalistic_target.pt')
        if os.path.isfile('observation.pt'):
            print("init from file")
            self.state_memory, self.new_state_memory, self.action_memory, self.reward_memory, self.terminal_memory = pickle.load(open('observation.pt', 'rb'))

        else:
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

    def choose_action(self, observation):
        obs_np = np.array([observation])
        if self.train:
            if np.random.random() > self.epsilon:

                state = torch.tensor(obs_np).to(self.Q_eval.device)
                actions = self.Q_eval.forward(state)
                action = torch.argmax(actions).item()
                # print('deep action happening')
            else:
                # print("here's random")
                action = np.random.choice(self.action_space)
        else:
            state = torch.tensor(obs_np).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
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

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        # q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0].detach()
        # q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        # print(q_target)

        loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.learn_step_cntr += 1

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end

    def replace_target_network(self):
        if self.replace is not None and self.learn_step_cntr % self.replace == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def save_model(self):
        # self.q_net.save(self.model_file)
        # torch.save(self.Q_eval, self.model_file)
        self.Q_eval.save_chkpt()
        self.Q_target.save_chkpt()

    def load_model(self, mode='gpu'):

        # self.q_net = load_model(self.model_file)
        # if not os.path.isfile(self.model_file):
        #     # np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
        #
        # else:
        self.Q_eval.load_chkpt()
        self.Q_target.load_chkpt()
        # if mode == 'gpu':
        #     # self.Q_eval.load_state_dict(torch.load(self.model_file))
        #     # self.Q_target.load_state_dict(torch.load(self.model_file))
        #     self.Q_eval = torch.load(self.model_file)
        #     self.Q_target = torch.load(self.model_file)
        # elif mode == 'cpu':
        #     self.Q_eval.load_state_dict(torch.load(self.model_file, map_location=torch.device('cpu')))
        #     self.Q_target.load_state_dict(torch.load(self.model_file, map_location=torch.device('cpu')))
        # else:
        #     raise Exception('Unsupported device entered. {} is not known. Please use either cpu or gpu'.format(mode))


def warmup(self, game_state):

    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

    def reset_self(self):
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        # While this timer is positive, agent will not hunt/attack opponents
        self.ignore_others_timer = 0

    def look_for_targets(free_space,start,targets,logger=None):
        if len(targets) == 0: return None

        frontier = [start]
        parent_dict = {start: start}
        dist_so_far = {start: 0}
        best = start
        best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

        while len(frontier) > 0:
            current = frontier.pop(0)
            # Find distance from current position to all targets, track closest
            d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
            if d + dist_so_far[current] <= best_dist:
                best = current
                best_dist = d + dist_so_far[current]
            if d == 0:
                # Found path to a target's exact position, mission accomplished!
                best = current
                break
            # Add unexplored free neighboring tiles to the queue in a random order
            x, y = current
            neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
            shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in parent_dict:
                    frontier.append(neighbor)
                    parent_dict[neighbor] = current
                    dist_so_far[neighbor] = dist_so_far[current] + 1
        if logger: logger.debug(f'Suitable target found at {best}')
        # Determine the first step towards the best found target tile
        current = best
        while True:
            if parent_dict[current] == start: return current
            current = parent_dict[current]

    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a