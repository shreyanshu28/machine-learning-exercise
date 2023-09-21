from collections import namedtuple, deque
import json

import pickle
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 30000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
REDUCED_DISTANCE_TO_NEXT_COIN = "REDUCED_DISTANCE_TO_COIN"
INCREASED_DISTANCE_TO_NEXT_COIN = "INCREASED_DISTANCE_TO_COIN"

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.alpha = 0.4
    self.gamma = 0.7
    self.total_step = 0


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
    def euclid(list, tuple):
        return [np.sqrt((list[i][0] - tuple[0])**2 + (list[i][1] - tuple[1])**2) for i in range(len(list))]
    
    if (self_action in ['UP', 'DOWN', 'LEFT', 'RIGHT']):
        oldPlayerLoc, newPlayerLoc = old_game_state['self'][3], new_game_state['self'][3]
        if len(old_game_state['coins']) != 0 and len(new_game_state['coins']) != 0:
            if min(euclid(old_game_state['coins'], oldPlayerLoc)) > min(euclid(new_game_state['coins'], newPlayerLoc)):
                events.append(REDUCED_DISTANCE_TO_NEXT_COIN)
            else:
                events.append(INCREASED_DISTANCE_TO_NEXT_COIN)
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    
    transition = Transition(state_to_features(self,old_game_state), self_action, state_to_features(self,new_game_state), reward_from_events(self, events))
    self.transitions.append(transition)
    self.total_step += 1
    if self.total_step % TRANSITION_HISTORY_SIZE == 0:
        for trans in self.transitions:
            update_Q(self,trans)
        with open("save_files/my-saved-model.json","w") as file:
            json.dump(self.Q, file)

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
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # print("end of round")
    #print(last_game_state[])
    # Store the model
    #if last_game_state['']:
    #print(type(self.Q))
    

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
        e.CRATE_DESTROYED: 0.4,
        e.MOVED_LEFT: 0.01,
        e.MOVED_RIGHT: 0.01,
        e.MOVED_UP: 0.01,
        e.MOVED_DOWN: 0.01,
        # e.INVALID_ACTION: -2,
        e.WAITED: -0.5,
        # e.GOT_KILLED: -1,
        e.KILLED_SELF: -3,
        e.COIN_COLLECTED: 10,
        REDUCED_DISTANCE_TO_NEXT_COIN: .2,
        INCREASED_DISTANCE_TO_NEXT_COIN: -.001
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def update_Q(self, Transition):
    if Transition[0] not in self.Q:
        self.Q[Transition[0]] = [0] * 6
    if Transition[2] not in self.Q:
        self.Q[Transition[2]] = [0] * 6

    
    action_index = ACTIONS.index(Transition[1])
    self.Q[Transition[0]][action_index] = (1-self.alpha)*self.Q[Transition[0]][action_index] + self.alpha*(Transition[3] + self.gamma*np.max(self.Q[Transition[2]]) - self.Q[Transition[0]][action_index])

### outdated

def add_other_transitions(self,old_game_state, new_game_state):

    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    old_me = old_game_state['self']
    new_me = new_game_state['self']
    type(new_me)
    old_others_name, old_others_score, old_others_bomb, old_others_xy = zip(*old_game_state['others'])
    new_others_name, new_others_score, new_others_bomb, new_others_xy = zip(*new_game_state['others'])
    #for i in range(len(old_others_name)):
    #    observation[others_x[i], others_y[i]] = -40 if others_bomb[i] else -30

    if old_game_state['others']:
        for i in range(len(old_others_name)):
            action = ""
            if old_others_name[i] in new_others_name:
                if old_others_bomb[i] > new_others_bomb[i]:
                    action = 'BOMB'
                elif old_others_xy[i][1] == new_others_xy[i][1]+1:
                    action = 'UP'
                elif old_others_xy[i][0] == new_others_xy[i][0]-1:
                    action = 'RIGHT'
                elif old_others_xy[i][1] == new_others_xy[i][1]-1:
                    action = 'DOWN'
                elif old_others_xy[i][0] == new_others_xy[i][0]+1:
                    action = 'LEFT'
                else:
                    action = 'WAIT'
                old_temp_switched = old_game_state
                new_temp_switched = new_game_state
                old_temp_switched['others'].append(old_me)
                new_temp_switched['others'].append(new_me)

                print(old_others_xy[i], new_others_xy[i], action)
                self.transitions.append()
                self.total_step += 1
            
    