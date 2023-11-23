import numpy as np
from os import listdir
import pickle
import requests
import json

GRID_WIDTH = 3  # Не менять
GRID_HEIGHT = 3  # Не менять
GRID_SIZE = GRID_WIDTH * GRID_HEIGHT
X = 'X'
O = 'O'
EMPTY = '_'
NUMBER_OF_GAMES = 10000
BASE_WIN_COUNTER = 10
BASE_LOSS_COUNTER = 0
BASE_Q = 0
EPSILON = 0
GAMMA = 1
ALPHA = 1
STEP_REWARD = -0.1
WIN_REWARD = 5
LOSS_REWARD = -3
DRAW_REWARD = -2
SHOW_GAME = 0
#O


class Table:
    def __init__(self):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.grid = np.array([[Cell(EMPTY, x, y) for x in range(self.width)] for y in range(self.height)])
        
    def __str__(self):
        result = ''
        for y in range(self.height):
            for x in range(self.width):
                result += self.grid[y][x].type + ' '
            result += '\n'
        return result
    
    def reset(self):
        self.grid = np.array([[Cell(EMPTY, x, y) for x in range(self.width)] for y in range(self.height)])
    
    def get_empty_cells(self):
        empty_cells = np.array([cell for cell in self.grid.ravel() if cell.type == EMPTY])
        return empty_cells
    
    def get_state(self):
        state = np.array([[cell.type for cell in row] for row in self.grid])
        return state
                
    def get_step_coords(self, state):
        for y, row in enumerate(state):
            for x, type in enumerate(row):
                if type != self.grid[y][x].type:
                    return x, y
               
    @staticmethod  
    def get_position_type(x, y):
        abs_dif_xy = abs(x - y)
        if abs_dif_xy == 1:
            return 'side'
        elif abs_dif_xy == 2:
            return 'corner_down-top-diag'
        elif x == 1:
            return 'middle'
        else:
            return 'corner_top-down-diag'
    

class Cell:
    def __init__(self, type, x, y):
        self.type = type
        self.x = x
        self.y = y  
 
 
class PrimalAgent:
    '''Random move playing'''
    
    def __init__(self, table, char, name='noname'):
        self.table = table
        self.char = char 
        self.name = name
        self.last_cell = None
        
    def move(self):
        empty_cells = self.table.get_empty_cells()
        target_cell = np.random.choice(empty_cells)
        self.table.grid[target_cell.y][target_cell.x].type = self.char
        self.last_cell = target_cell
        

class RemoteAgent(PrimalAgent):
    def __init__(self, table, char, url="http://45.130.43.227:5000/api", name='noname'):
        super().__init__(table, char, name)
        self.url = url
        
    def move(self):
        state = self.table.get_state().tolist()
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.url, data=json.dumps(state), headers=headers)   
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
        state = np.array(json.loads(response.text))
        x, y = self.table.get_step_coords(state)
        self.table.grid[y][x].type = self.char
        self.last_cell = self.table.grid[y][x]
 
 
class RealPlayer:
    def __init__(self, table, char):
        self.char = char
        self.table = table
        self.last_cell = None
    
    def move(self):
        coords = input('Введите координаты в формате XY:\n\n')
        print()
        x = int(coords[0]) - 1
        y = int(coords[1]) - 1
        cell = self.table.grid[y][x]
        cell.type = self.char
        self.last_cell = cell
  
 
# class StateNode:
#     def __init__(self, hash, win_counter=BASE_WIN_COUNTER, loss_counter=BASE_LOSS_COUNTER):
#         self.hash = hash
#         self.next_states = []
#         self.win_counter = win_counter
#         self.loss_counter = loss_counter
        
        
class QAgent(PrimalAgent):
    '''Q learning agent'''

    def __init__(self, table, char, game, name='noname', epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, 
                 step_reward=STEP_REWARD, win_reward=WIN_REWARD, loss_reward=LOSS_REWARD, 
                 draw_reward=DRAW_REWARD, base_win_counter=BASE_WIN_COUNTER, base_loss_counter=BASE_LOSS_COUNTER):
        super().__init__(table, char, name)
        self.game = game
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.step_reward = step_reward
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.drow_reward = draw_reward
        self.base_win_counter = base_win_counter
        self.base_loss_counter = base_loss_counter
        self.base_Q = base_win_counter / base_loss_counter
        self.actions = []
        if f'{self.name}_q_table' in globals():
            self.q_table = globals()[f'{self.name}_q_table']
        elif f'{self.name}_q_table.pkl' in listdir():
            with open(f'{self.name}_q_table.pkl', 'rb') as f:
                self.q_table = pickle.load(f)
            globals()[f'{self.name}_q_table'] = self.q_table
        else:
            self.q_table = dict()
            globals()[f'{self.name}_q_table'] = self.q_table
        print(len(self.q_table))
        
    def move(self):
        state = self.table.get_state()
        if self.char == O:
            state = self.mirror_state(state)
        action_x, action_y = self.step(state)
        self.table.grid[action_y][action_x].type = self.char
        self.last_cell = self.table.grid[action_y][action_x]
        
    def update_q_table(self, result):
        for action in self.actions:
            action[result] += 1
            action['Q'] = action['win'] / action['loss']
        self.actions = []
        
    def step(self, state):
        def get_coords(action_state):
            for y in range(self.table.height):
                for x in range(self.table.width):
                    if action_state[y][x] == '.':
                        return x, y 
            
        action_state = self.get_action(state)
        action_x, action_y = get_coords(action_state)        
        return action_x, action_y
    
    def get_Q_max(self):
        if self.game.move_counter == 9:
            return BASE_Q
        state = self.table.get_state()
        hash, num = self.find_or_create_state_in_q_table(state)
        # print(hash)
        Q_max = max(self.q_table[hash].values())
        return Q_max
    
    def get_state_by_action_state(self, action_state):
        for y in range(self.table.height):
            for x in range(self.table.width):
                if action_state[y][x] == '.':
                    action_state[y][x] == X
                    return action_state
    
    def get_action(self, state):
        dirty_state_hash, num = self.find_or_create_state_in_q_table(state)
        self.last_dirty_state_hash = dirty_state_hash
        if np.random.random() < self.epsilon:
            dirty_action_hash = np.random.choice(list(self.q_table[dirty_state_hash].keys()))
        else:
            if SHOW_GAME:
                print(self.q_table[dirty_state_hash])
            max_q = max([action['Q'] for action in self.q_table[dirty_state_hash]])
            dirty_action_hash = np.random.choice([action for action in self.q_table[dirty_state_hash] if action['Q'] == max_q])
        # print(Q_max)
        self.actions.append(self.q_table[dirty_state_hash][dirty_action_hash])
        action_state = self.reverse_hash_and_get_state(dirty_action_hash, num)
        return action_state
        
    def find_or_create_state_in_q_table(self, state):
        for num, similar_state in enumerate(self.similar_states_generator(state.copy())):
            hash = self.get_hash(similar_state)
            if hash in self.q_table:
                similar_state
                return hash, num
        return self.add_hash_to_q_table(state), 0
         
    def add_hash_to_q_table(self, state):  
        hash = self.get_hash(state)  
        self.q_table[hash] = self.get_actions(state)
        return hash
    
    def get_actions(self, state):
        actions = {}
        for row in state:
            for key, type in enumerate(row):
                if type == EMPTY:
                    row[key] = '.'
                    actions[self.get_hash(state)] = {'win': self.base_win_counter, 
                                                     'loss': self.base_loss_counter, 
                                                     'Q': self.base_Q}
                    row[key] = EMPTY
        if not actions:
            actions['end'] = BASE_Q
            
        return actions
    
    def reverse_hash_and_get_state(self, hash, num):
        state = self.get_state_from_hash(hash)
        if num > 3:
            state = np.rot90(state, 4 - num)
            state = np.fliplr(state)
            state = np.rot90(state, -3)
        else:
            state = np.rot90(state, -num)
        return state
    
    def save_q_table(self):
        with open(f'{self.name}_q_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def mirror_state(self, state):
        for row in state:
            for key, type in enumerate(row):
                if type == X:
                    row[key] = O
                elif type == O:
                    row[key] = X
        return state
    
    @staticmethod
    def get_state_from_hash(hash):
        # print(hash)
        state = np.array(list(hash)).reshape(GRID_WIDTH, GRID_HEIGHT)
        return state
    
    @staticmethod   
    def get_hash(state):
        return "".join(state.ravel())
         
    @staticmethod    
    def similar_states_generator(state):
        yield state
        for k in range(7):
            if k == 3:
                state = np.fliplr(state)
            else:
                state = np.rot90(state)
            yield state
        
   
class Game:
    def __init__(self):
        self.table = Table()                                     #  RealPlayer(self.table, X)  RemoteAgent(self.table, X) 
        self.player_X = QAgent(self.table, X, self, name='Z')     #  QAgent(self.table, X, self, name='X')    PrimalAgent(self.table, X)
        self.player_O = QAgent(self.table, O, self, name='H')    # PrimalAgent(self.table, O)  QAgent(self.table, O, self, name='O')
        self.players = [self.player_X, self.player_O]
        self.winner = None
        self.move_counter = 0
        self.drow_counter = 0
        self.win_X_counter = 0
        self.win_O_counter = 0
        self.need_check_win = True
        
    def play(self):
        for _ in range(NUMBER_OF_GAMES):
            self.cicle()
            self.table.reset()
            self.winner = None
            self.players = self.players[::-1]
            self.move_counter = 0
            
        print(f'Draw {self.drow_counter}')
        print(f'WinX {self.win_X_counter}')
        print(f'WinO {self.win_O_counter}')
        
        for player in self.players:
            if player.__class__ is QAgent and f'{player.name}_q_table' in globals():
                player.save_q_table()
                del globals()[f'{player.name}_q_table']
        
    def cicle(self):
        if SHOW_GAME:
            print(self.table)
        while True:
            for player in self.players:
                self.move_counter += 1
                player.move()
                if SHOW_GAME:
                    print(self.table)
                if self.is_win(player.last_cell.x, player.last_cell.y):
                    self.winner = player
                    break
                if self.move_counter == GRID_SIZE:
                    break
            else:
                continue
            break
        self.update_q_tables()
        self.game_over()
        
    def game_over(self):
        if self.winner:
            if SHOW_GAME:
                print(f'Winner is {self.winner.char}\n')
            if self.winner is self.player_X:
                self.win_X_counter += 1
            else:
                self.win_O_counter += 1
        else:
            if SHOW_GAME:
                print('Draw\n')
            self.drow_counter += 1
        
    def update_q_tables(self):
        if self.winner:
            for player in self.players:
                if player.__class__ is QAgent:
                    if self.winner is player:
                        result = 'win'
                    else:
                        result = 'loss'
                player.update_q_table(result=result)
        
    def is_win(self, x, y):
        def win_by_row(self, y):
            row = set([self.table.grid[y][x].type for x in range(self.table.width)])
            win = len(row) == 1
            return win
        
        def win_by_col(self, x):
            col = set([self.table.grid[y][x].type for y in range(self.table.height)])
            win = len(col) == 1
            return win
        
        def win_by_top_down_diag(self):
            diag = set([self.table.grid[y][y].type for y in range(self.table.height)])
            win = len(diag) == 1
            return win
        
        def win_by_down_top_diag(self):
            diag = set([self.table.grid[abs(y-2)][y].type for y in range(self.table.height)])
            win = len(diag) == 1
            return win
        
        def win_by_both_diags(self):
            win = win_by_top_down_diag(self) or win_by_down_top_diag(self)
            return win
        
        if self.move_counter < 5:
            return False
        if win_by_row(self, y) or win_by_col(self, x):
            return True
        position_type = self.table.get_position_type(x, y)
        if position_type == 'corner_top-down-diag':
            if win_by_top_down_diag(self):
                return True
        elif position_type == 'corner_down-top-diag':
            if win_by_down_top_diag(self):
                return True
        elif position_type == 'middle':
            if win_by_both_diags(self):
                return True
        else:
            return False
        
         
    

game = Game()
game.play()
    

         