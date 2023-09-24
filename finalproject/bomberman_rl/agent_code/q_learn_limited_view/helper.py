import numpy as np

BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2

def euclid(list, tuple):
    return [np.sqrt((list[i][0] - tuple[0])**2 + (list[i][1] - tuple[1])**2) for i in range(len(list))]

def calc_dir(list, tuple):
    shortest_dist_index = np.argmin(euclid(list,tuple))
    x_val = tuple[0] - list[shortest_dist_index][0]
    #print("x_val: ", x_val, tuple[0], x_val < 0)
    
    y_val = tuple[1] - list[shortest_dist_index][1]
    #print("y_val: ", y_val, tuple[1], y_val < tuple[1])    
    return abs(x_val) + abs(y_val)


def get_blast_coords(x, y, arena):
    blast_coords = [(x, y)]

    for i in range(1, BOMB_POWER + 1):
        if arena[x + i, y] == -1:
            break
        blast_coords.append((x + i, y))
    for i in range(1, BOMB_POWER + 1):
        if arena[x - i, y] == -1:
            break
        blast_coords.append((x - i, y))
    for i in range(1, BOMB_POWER + 1):
        if arena[x, y + i] == -1:
            break
        blast_coords.append((x, y + i))
    for i in range(1, BOMB_POWER + 1):
        if arena[x, y - i] == -1:
            break
        blast_coords.append((x, y - i))

    return blast_coords