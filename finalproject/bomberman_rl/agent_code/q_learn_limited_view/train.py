from collections import namedtuple, deque
import json

import pickle
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features
from . import helper as h

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 2000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
REDUCED_DISTANCE_TO_NEXT_COIN_EVENT = "REDUCED_DISTANCE_TO_COIN"
INCREASED_DISTANCE_TO_NEXT_COIN_EVENT = "INCREASED_DISTANCE_TO_COIN"
REDUCED_DISTANCE_TO_ENEMY_EVENT = "REDUCED_DISTANCE_TO_ENEMY"
INCREASED_DISTANCE_TO_BOMB_EVENT = "INCREASED_DISTANCE_TO_BOMB"
NEW_TILE_FOUND = "DISCOVERED_NEW_TILE"
WALKED_OUT_OF_EXPLOSION_EVENT = "WALKED_OUT_OF_EXPLOSION"
ON_DANGEROUS_FIELD_EVENT = "ON_DANGEROUS_FIELD"
WAITED_ON_DANGEROUS_FIELD_EVENT = "WAITED_ON_DANGEROUS_FIELD"

DUMB_BOMB_EVENT = "DUMB_BOMB"
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
    self.walkedTiles = []
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.alpha = 0.5
    self.gamma = 0.1
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

    # state_to_features is defined in callbacks.py

    events = add_events(self, old_game_state, self_action, new_game_state, events)
    transition = Transition(state_to_features(self,old_game_state), self_action, state_to_features(self,new_game_state), reward_from_events(self, events))
    self.transitions.append(transition)
    self.total_step += 1
    if self.total_step % TRANSITION_HISTORY_SIZE == 0:
        for trans in self.transitions:
            update_Q(self,trans)
        with open(self.file_path,"wb") as file:
            pickle.dump(self.Q, file)

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
    events = add_events(self, last_game_state,last_action, None, events)
    transition = Transition(state_to_features(self,last_game_state), last_action, None, reward_from_events(self, events))
    self.transitions.append(transition)
    self.total_step += 1
    if self.total_step % TRANSITION_HISTORY_SIZE == 0:
        for trans in self.transitions:
            update_Q(self,trans)
        with open(self.file_path, "wb") as file:
            pickle.dump(self.Q, file)
    
def update_Q(self, Transition):
    if Transition[0] not in self.Q:
        self.Q[Transition[0]] = [0] * 6
    if Transition[2] not in self.Q:
        self.Q[Transition[2]] = [0] * 6

    
    action_index = ACTIONS.index(Transition[1])
    self.Q[Transition[0]][action_index] = ((1-self.alpha)*self.Q[Transition[0]][action_index])\
                                            + self.alpha*(Transition[3] + self.gamma*np.max(self.Q[Transition[2]]))


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION: -20,
        # e.COIN_COLLECTED: 1,
        #e.KILLED_OPPONENT: 50,
        e.BOMB_DROPPED: 1,
        #e.COIN_FOUND: 10,
        #e.SURVIVED_ROUND: 0.5,
        e.CRATE_DESTROYED: 0.5,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        # e.INVALID_ACTION: -2,
        e.WAITED: -2,
        #e.GOT_KILLED: -100,
        e.KILLED_SELF: -2,
        #e.COIN_COLLECTED: 10,
        DUMB_BOMB_EVENT: -2,
        NEW_TILE_FOUND: 1,
        #REDUCED_DISTANCE_TO_NEXT_COIN_EVENT: 1,
        #INCREASED_DISTANCE_TO_NEXT_COIN_EVENT: -.001,
        INCREASED_DISTANCE_TO_BOMB_EVENT: 1,
        #ON_DANGEROUS_FIELD_EVENT: -1,
        #WAITED_ON_DANGEROUS_FIELD_EVENT: -2,
        #WALKED_OUT_OF_EXPLOSION_EVENT: 10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def add_events(self, old_game_state,self_action, new_game_state, events):
    if old_game_state == None or new_game_state == None:
        return events
    # Idea: Add your own events to hand out rewards
    bombs = []
    if len(old_game_state['bombs']) != 0:
        bombs = [xy for (xy,t) in old_game_state['bombs']]
    
    if self_action == 'BOMB' and old_game_state['self'][3] in [(1,1),(1,16),(16,1),(16,16)]:
        events.append(DUMB_BOMB_EVENT)
    if (self_action in ['UP', 'DOWN', 'LEFT', 'RIGHT']):
        oldPlayerLoc, newPlayerLoc = old_game_state['self'][3], new_game_state['self'][3]
        if len(old_game_state['coins']) != 0 and len(new_game_state['coins']) != 0:
            if min(h.euclid(old_game_state['coins'], oldPlayerLoc)) > min(h.euclid(new_game_state['coins'], newPlayerLoc)):
                events.append(REDUCED_DISTANCE_TO_NEXT_COIN_EVENT)
            else:
                events.append(INCREASED_DISTANCE_TO_NEXT_COIN_EVENT)

        if len(old_game_state['others']) != 0:
                others = [xy for (n, s, b, xy) in old_game_state['others']]
                if min(h.euclid(others, oldPlayerLoc)) >= min(h.euclid(others, newPlayerLoc)):
                    events.append(REDUCED_DISTANCE_TO_ENEMY_EVENT)

        if new_game_state['self'][3] not in self.walkedTiles:
            self.walkedTiles.append(new_game_state['self'][3])
            events.append(NEW_TILE_FOUND)
        if bombs:
            if min(h.euclid(bombs, oldPlayerLoc)) < min(h.euclid(bombs, newPlayerLoc)):
                events.append(INCREASED_DISTANCE_TO_BOMB_EVENT)

    else:
        oldPlayerLoc = old_game_state['self'][3]
        newPlayerLoc = new_game_state['self'][3]

    if new_game_state == None:
        self.walkedTiles = []

    expl_coord = []
    
    for bomb in bombs:
        expl_coord.extend(h.get_blast_coords(bomb[0], bomb[1], old_game_state['field']))
    if newPlayerLoc in expl_coord:
        if self_action == 'WAIT':
            events.append(WAITED_ON_DANGEROUS_FIELD_EVENT)
        events.append(ON_DANGEROUS_FIELD_EVENT)
    if oldPlayerLoc in expl_coord and newPlayerLoc not in expl_coord:
        events.append(WALKED_OUT_OF_EXPLOSION_EVENT)
        
        


    return events
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
            
    