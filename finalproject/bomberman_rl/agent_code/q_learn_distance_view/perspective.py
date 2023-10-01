from collections import deque
import numpy as np
from . import hyperparameter as hp

def get_manhattan_dist(koord1, koord2):
    # Compute the manhattan distance
    return np.abs(koord1[0]-koord2[0]) + np.abs(koord1[1]-koord2[1])

# Create input values for neural network by looking in every possible direction
def state_to_features(game_state):
    #Returns a 4 x distance matrix containing the events in the line of sight in every direction until distance 

    result = np.zeros((4, 6))
    x_self, y_self = game_state['self'][3]

    map = game_state['field']
    cols = len(map[0])
    rows = len(map)
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosions = game_state['explosion_map']
    others = game_state['others']

    for k, pos in enumerate([(-1,0),(1,0),(0,1),(0,-1)]):
        x_pos, y_pos = pos

        # If wall is in this direction fill whole entry with wall
        if map[x_self + x_pos][y_self + y_pos] == -1:
            result[k] = np.full_like(result[k], hp.WALL)
            continue

        # Distance to next bomb in this direction
        bombdist = -1
        for bomb in bombs:
            dist = get_manhattan_dist((x_self + x_pos, y_self + y_pos), bomb[0])
            if dist < bombdist or bombdist == -1:
                bombdist = dist

        result[k,1] = bombdist

        # Countdown until next field in this direction is dangerous
        if explosions[x_self + x_pos][y_self + y_pos] == 0 and bombdist > 3: # No current explosion and not in radius of unexploded bomb
            badtimer = -1
        elif explosions[x_self + x_pos][y_self + y_pos] > 0: # Current explosion 
            badtimer = 0
        else: 
            badtimer = -1
            for bomb in bombs:
                if (bomb[0][0] == x_self+x_pos and np.abs(bomb[0][1]-y_self+y_pos) < 5) or (bomb[0][1] == y_self+y_pos and np.abs(bomb[0][0]-x_self+x_pos) < 5):
                    if badtimer == -1 or bomb[1] < badtimer:
                        badtimer = bomb[1]

        result[k,2] = badtimer



        # Distance to next coin in this direction
        coindist = -1
        for coin in coins:
            dist = get_manhattan_dist((x_self + x_pos, y_self + y_pos), coin)
            if dist < coindist or coindist == -1:
                coindist = dist

        result[k,3] = coindist


        # Distance to next crate in this direction
        indices = np.where(np.array(map) == 1)
        index_pairs = list(zip(indices[0], indices[1]))
        cratedist = -1
        for crate in index_pairs:
            dist = get_manhattan_dist((x_self + x_pos, y_self + y_pos), crate)
            if dist < cratedist or cratedist == -1:
                cratedist = dist

        result[k,4] = cratedist


        # Distance to next enemy in this direction
        enemydist = -1
        for o in others:
            dist = get_manhattan_dist((x_self + x_pos, y_self + y_pos), o[3])
            if dist < enemydist or enemydist == -1:
                enemydist = dist


        result[k,5] = enemydist

    # Flatten matrix
    result = result.reshape(-1)


    # Countdown until current field becomes dangerous
    # No need to check for current explosion. Agent would be already dead!
    self_countdown = -1

    for j in range(4):
        for bomb in bombs:
            # Look if bomb is in straight line from agent and therefore dangerous
            if (bomb[0][0] == x_self and np.abs(bomb[0][1]-y_self) < 5) or (bomb[0][1] == y_self and np.abs(bomb[0][0]-x_self) < 5):
                if badtimer == -1 or bomb[1] < badtimer:
                    badtimer = bomb[1]

    # Boolean if bomb is droppable
    bomb_value = np.float64(game_state['self'][2])

    append_part = [[self_countdown, bomb_value]]

    result = np.append(result, append_part)
    result = result.reshape(1,26)

    return tuple(result.tostring()).__hash__()
