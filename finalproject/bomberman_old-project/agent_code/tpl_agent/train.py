from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 5  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "REDUCED_DISTANCE_TO_COIN"
PLACEHOLDER_EVENT = "REDUCED_DISTANCE_TO_ENEMY"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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

    # Idea: Add your own events to hand out rewards
    # Here are condition for triggereing event
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -50,
        e.SURVIVED_ROUND:
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

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

	return observation


class ReplayBuffer:
	def __init__(self, mem_size, input_dims):
		self.mem_size = mem_size
		self.mem_cntr = 0

		self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

	# state_ is next state
	def store_transition(self, state, action, reward, state_):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.mem_cntr += 1

	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, batch_size, replace=False)

		states = self.state_memory[batch]
		states_ = self.new_state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]

		return states, actions, rewards, states_, terminal

	def get_last(self, n):



# def build_dqn(lr, n_actions, input_dims):
#			inputs = tf.keras.Input(shape=input_dims)
#			x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
#			x = tf.keras.layers.Flatten()(x)
#			x = tf.keras.layers.Dense(256, activation='relu')(x)
#			x = tf.keras.layers.Dense(256, activation='relu')(x)
#			outputs = tf.keras.layers.Dense(n_actions, activation=None)(x)
#			model = tf.keras.Model(inputs=inputs, outputs=outputs)
#
#			model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
#			return model

class DQNModel(nn.Sequential):

	def __init__(self, n_actions, input_dims):
		self.n_actions = n_actions
		super(DQNModel, self).__init__()
		self.layer1 = nn.Conv2d(32, 16, kernel_size=(4, 4), padding='same')
		self.layer2 = nn.Linear(256, 256)
		self.layer3 = nn.Linear(256, 256)
		self.output = nn.Linear(256, n_actions)

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = torch.flatten(x, 1)
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		return x.view(-1, self.n_actions)


class DQNAgent:
	def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
				 input_dims, epsilon_dec=1e-3, epsilon_end=0.01,
				 mem_size=1000000, fname='dqn_model_bombalistic.h5'):
		self.action_space = [i for i in range(n_actions)]
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_dec = epsilon_dec
		self.eps_min = epsilon_end
		self.batch_size = batch_size
		self.model_file = fname
		self.memory = ReplayBuffer(mem_size, input_dims)
		# self.q_net = build_dqn(lr, n_actions, input_dims)
		self.q_net = DQNModel(n_actions, input_dims)
		self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
		self.loss_fn = nn.SmoothL1Loss()

	def store_transition(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)

	def learn(self):
		if self.memory.mem_cntr < self.batch_size:
			# only learn if buffer is full enough
			return

		states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

		q_eval = self.q_net(states)
		q_next = self.q_net(states_)

		q_target = np.copy(q_eval)
		batch_index = np.arange(self.batch_size, dtype=np.int32)

		q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * dones

		# self.q_net.train_on_batch(states, q_target)
		loss = self.loss_fn(q_eval, q_next)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		loss = loss.item()

		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

	def choose_action(self, observation):
		# TODO: temporarily disabled exploration to debug NN
		# if np.random.random() < self.epsilon:
		# 	action = np.random.choice(self.action_space)
		# else:
		print(observation[np.newaxis, :])
		actions = self.q_net(observation[np.newaxis, :])
		action = np.argmax(actions)

		return ACTIONS[action]

	def save_model(self):
		# self.q_net.save(self.model_file)
		torch.save(self.q_net.state_dict(), self.model_file)

	def load_model(self):
		# self.q_net = load_model(self.model_file)
		# if not os.path.isfile(self.model_file):
		#     # np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
		#
		# else:
		if os.path.isfile(self.model_file):
			self.q_net.load_state_dict(torch.load(self.model_file))
