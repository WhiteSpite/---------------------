from os import listdir
from sklearn.cluster import KMeans
import numpy as np
import pickle
import requests
import json

from config import *


WIN_REWARD = 1
LOSS_REWARD = -1
DRAW_REWARD = 0

BASE_Q = 2
EPSILON = 0
ALPHA = 1
ALPHA_DIVIDER = 1

BASE_WIN_COUNTER = 1
BASE_LOSS_COUNTER = 1
BASE_DRAW_COUNTER = 1

STOP_LEARNING = 1
TO_CLUSTER = 0
 
 
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
        x, y = self.table.get_action_coords(state)
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
        
                    
class QAgent(PrimalAgent):
    '''Q learning agent'''

    def __init__(self, table, char, game, name='noname', epsilon=EPSILON, win_reward=WIN_REWARD, loss_reward=LOSS_REWARD, 
                 draw_reward=DRAW_REWARD, stop_learning=STOP_LEARNING, alpha=ALPHA, alpha_divider=ALPHA_DIVIDER):
        super().__init__(table, char, name)
        self.game = game
        self.epsilon = epsilon
        self.alpha = alpha
        self.alpha_divider = alpha_divider
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.draw_reward = draw_reward
        self.base_Q = BASE_Q
        self.stop_learning = stop_learning
        self.actions = []
        self._init_q_table()
        
    def _init_q_table(self):
        if f'{self.name}_q_table' in globals():
            self.q_table = globals()[f'{self.name}_q_table']
        elif f'{self.name}_q_table.pkl' in listdir():
            with open(f'{self.name}_q_table.pkl', 'rb') as f:
                self.q_table = pickle.load(f)
            globals()[f'{self.name}_q_table'] = self.q_table
        else:
            self.q_table = {'clustered': False}
            globals()[f'{self.name}_q_table'] = self.q_table  
        
    def move(self):
        state = self.table.get_state()
        if self.char == O:
            state = self.mirror_state(state)
        action_x, action_y = self.step(state)
        self.table.grid[action_y][action_x].type = self.char
        self.last_cell = self.table.grid[action_y][action_x]
        
    def update_q_table(self, result):
        reward = self.__dict__[f'{result}_reward']
        for action in self.actions:
            if action['alpha'] > 0:
                td_error = reward - action['Q']
                action['Q'] += td_error * action['alpha']
                action['alpha'] /= self.alpha_divider
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
    
    def get_action(self, state):
        dirty_state_hash, num = self.find_or_create_state_in_q_table(state)
        self.last_dirty_state_hash = dirty_state_hash
        if np.random.random() < self.epsilon:
            dirty_action_hash = np.random.choice(list(self.q_table[dirty_state_hash].keys()))
        else:
            if self.q_table['clustered']:
                Q = 'clustered_Q'
            else:
                Q = 'Q'
            max_q = max([self.q_table[dirty_state_hash][action][Q] for action in self.q_table[dirty_state_hash]])
            dirty_action_hash = np.random.choice([action for action in self.q_table[dirty_state_hash] \
                if self.q_table[dirty_state_hash][action][Q] == max_q])
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
                    actions[self.get_hash(state)] = {'alpha': self.alpha,
                                                     'Q': self.base_Q}
                    row[key] = EMPTY
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
        if f'{self.name}_q_table' in globals() and not self.stop_learning:
            with open(f'{self.name}_q_table.pkl', 'wb') as f:
                pickle.dump(self.q_table, f)
            del globals()[f'{self.name}_q_table']
        
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
            

class MeanQAgent(QAgent):
    '''Arithmetic mean (wins, losses, draws) learning agent'''
    
    def __init__(self, table, char, game, name='noname', epsilon=EPSILON, win_reward=WIN_REWARD, loss_reward=LOSS_REWARD, 
                 draw_reward=DRAW_REWARD, stop_learning=STOP_LEARNING, base_win_counter=BASE_WIN_COUNTER, 
                 base_loss_counter=BASE_LOSS_COUNTER, base_draw_counter=BASE_DRAW_COUNTER, to_cluster=TO_CLUSTER):
        super().__init__(table, char, game, name, epsilon, win_reward, loss_reward, draw_reward, stop_learning)
        self.base_win_counter = base_win_counter
        self.base_loss_counter = base_loss_counter
        self.base_draw_counter = base_draw_counter
        self.to_cluster = to_cluster
        self._cluster_q_table()
    
    def _cluster_q_table(self):
        if self.stop_learning and self.to_cluster:
            new_Q = []
            all_actions = []
            n_clasters = 20
            for hash in self.q_table:
                if type(self.q_table[hash]) is not bool:
                    for action in self.q_table[hash]:
                        all_actions.append(self.q_table[hash][action])
                        new_Q.append(self.q_table[hash][action]['Q'])
            X = np.array(new_Q).reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clasters).fit(X)
            labels = kmeans.labels_
            indexes = np.argsort(kmeans.cluster_centers_.squeeze())
            lookup_table = np.zeros_like(indexes)
            lookup_table[indexes] = np.arange(n_clasters)
            ordered_labels = lookup_table[labels]
            for action, label in zip(all_actions, ordered_labels):
                action['clustered_Q'] = label
            self.q_table['clustered'] = True 
    
    def update_q_table(self, result):
        for action in self.actions:
            action[result] += 1
            action['Q'] = (action['win'] * self.win_reward \
                         + action['loss'] * self.loss_reward \
                         + action['draw'] * self.draw_reward) \
                         / (action['win'] + action['loss'] + action['draw'])
        self.actions = []
        
    def get_actions(self, state):
        actions = {}
        for row in state:
            for key, type in enumerate(row):
                if type == EMPTY:
                    row[key] = '.'
                    actions[self.get_hash(state)] = {'win': self.base_win_counter,
                                                     'loss': self.base_loss_counter,
                                                     'draw': self.base_draw_counter,
                                                     'Q': self.base_Q}
                    row[key] = EMPTY
        return actions
    
    def save_q_table(self):
        if f'{self.name}_q_table' in globals() and (self.to_cluster or not self.stop_learning):
            with open(f'{self.name}_q_table.pkl', 'wb') as f:
                pickle.dump(self.q_table, f)
            del globals()[f'{self.name}_q_table']
