import numpy as np 
from time import time

class game_2048:

    def __init__(self, grid_shape = (4, 4), p2 = .8):
        self.grid_shape = grid_shape
        self.p = [p2, 1 - p2]
        self.actions = {0 : "left", 1 : "right", 2 : "up", 3 : "down"}
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_shape)
        self.grid[self.pick_cell(self.grid)] = np.random.choice([.1, .2], size = 1, p = self.p)
        self.next_states = self.get_moves()

    def get_moves(self):
        transitions = {}
        for mov in [0, 1, 2, 3]:
            aux = self.merge_matrix(self.grid, self.actions[mov])
            if (aux != self.grid).any(): transitions[mov] = aux 
        return transitions

    def update(self, action):
        game_over = False
        reward = 0
        self.grid = self.next_states[action]
        self.next_states = self.get_moves()
        if self.next_states == {}:
            game_over = True
            reward = -1
        elif (self.grid == 1).any():
            game_over = True
            reward = 1
        return self.grid.copy(), reward, game_over

    def get_state(self):
    	return self.grid.flatten()[np.newaxis]

    def merge_matrix(self, mat, mov):
        
        def merge_row_right(row):
            row_merge = np.zeros(row.shape)
            nonzero_cells = row[np.where(row != 0)]
            if nonzero_cells.size == 0: return row
            row_merge[-nonzero_cells.shape[0]:] = nonzero_cells
            for i in np.arange(row_merge.shape[0] - nonzero_cells.shape[0] + 1, 
                                       row_merge.shape[0])[::-1]:
                if row_merge[i] == row_merge[i-1]:
                    row_merge[i] += .1
                    row_merge[:i] = np.append([0], row_merge[:i-1])

            return row_merge

        mat_c = mat.copy()
        if mov == "right":
            for i, row in enumerate(mat_c):
                mat_c[i] = merge_row_right(row)
        elif mov == "left":
            mat_c = mat_c[:, ::-1]
            for i, row in enumerate(mat_c):
                mat_c[i] = merge_row_right(row)
            mat_c = mat_c[:, ::-1]
        elif mov == "up":
            mat_c = mat_c.T[:, ::-1]
            for i, row in enumerate(mat_c):
                mat_c[i] = merge_row_right(row)
            mat_c = mat_c[:, ::-1].T
        elif mov == "down":
            mat_c = mat_c.T
            for i, row in enumerate(mat_c):
                mat_c[i] = merge_row_right(row)
            mat_c = mat_c.T

        if (mat_c != mat).any():
            mat_c[self.pick_cell(mat_c)] = np.random.choice([.1, .2], size = 1, p = self.p)
        return mat_c

    def pick_cell(self, mat):
        zero_cells = np.argwhere(mat == 0)
        coords = zero_cells[np.random.choice(zero_cells.shape[0])]
        return tuple(coords)

'''
env = game_2048()
M = np.array([[.1, .2, 0, .2],
              [0, 0, 0, 0],
              [.4, .4, .4, .4,],
              [.1, .2, .1, 0]])
print(env.grid)
for k, v in env.get_moves().items():
    print(str(k) + ":")
    print(v)

t1 = time()
env.get_moves()
t2 = time()
print(t2 - t1)
'''