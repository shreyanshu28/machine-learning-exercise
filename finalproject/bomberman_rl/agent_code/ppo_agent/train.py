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
        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), self.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros(self.num_steps,self.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros(self.num_steps,self.num_envs)).to(device)
        rewards = torch.zeros(self.num_steps,self.num_envs)).to(device)
        dones = torch.zeros(self.num_steps,self.num_envs)).to(device)
        values = torch.zeros(self.num_steps,self.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zerosself.num_envs).to(device)
        num_updates =self.total_timesteps //self.batch_size

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac *self.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0,self.num_steps):
                global_step += 1 *self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                for item in info:
                    if "episode" in item.keys():
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t ==self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] +self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta +self.gamma *self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arangeself.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0,self.batch_size,self.minibatch_size):
                    end = start +self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() >self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 -self.clip_coef, 1 +self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            self.clip_coef,
                        self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss -self.ent_coef * entropy_loss + v_loss *self.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(),self.max_grad_norm)
                    optimizer.step()

                if self.target_kl is not None:
                    if approx_kl >self.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


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



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)




