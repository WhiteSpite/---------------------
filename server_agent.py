import numpy as np
from os import listdir
from flask import Flask, request, jsonify
import pickle

GRID_WIDTH = 3  # Не менять
GRID_HEIGHT = 3  # Не менять
GRID_SIZE = GRID_WIDTH * GRID_HEIGHT
X = 'X'
O = 'O'
EMPTY = '_'
NUMBER_OF_GAMES = 100000
BASE_Q = 1
EPSILON = 0
GAMMA = 0.9
ALPHA = 0.001
STEP_REWARD = 0
WIN_REWARD = 1
LOSE_REWARD = -1
DRAW_REWARD = 0
POLICY = 'egreedy' # 'egreedy' or 'greedy'
SHOW_GAME = 0


class Table:
    def __init__(self, state):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.grid = np.array([[Cell(EMPTY, x, y) for x in range(self.width)] for y in range(self.height)])
        for y, row in enumerate(state):
            for x, type in enumerate(row):
                self.grid[y][x].type = type
        
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
    
    def get_X_cells(self):
        empty_cells = np.array([cell for cell in self.grid.ravel() if cell.type == X])
        return empty_cells
    
    def get_O_cells(self):
        empty_cells = np.array([cell for cell in self.grid.ravel() if cell.type == O])
        return empty_cells
    
    def get_state(self):
        state = np.array([[cell.type for cell in row] for row in self.grid])
        return state
               
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
    
    def __init__(self, table, char):
        self.table = table
        self.char = char 
        self.last_cell = None
        
    def move(self):
        empty_cells = self.table.get_empty_cells()
        target_cell = np.random.choice(empty_cells)
        self.table.grid[target_cell.y][target_cell.x].type = self.char
        self.last_cell = target_cell
        
        
class QAgent(PrimalAgent):
    '''Q learning agent'''

    def __init__(self, table, char, game, name='noname', epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, 
                 step_reward=STEP_REWARD, win_reward=WIN_REWARD, lose_reward=LOSE_REWARD, 
                 draw_reward=DRAW_REWARD, policy=POLICY):
        super().__init__(table, char)
        self.game = game
        self.name = name
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.step_reward = step_reward
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.drow_reward = draw_reward
        self.policy = policy
        self.last_state_hash = None
        self.last_action_hash = None
        self.last_Q_max = None
        if f'{self.name}_q_table.pkl' in listdir():
            with open(f'{self.name}_q_table.pkl', 'rb') as f:
                self.q_table = pickle.load(f)
        else:
            self.q_table = dict()
        
    def move(self):
        state = self.table.get_state()
        state = self.mirror_state_by_char(state)
        action_x, action_y, Q_max, state_hash, dirty_action_hash = self.step(state)
        self.table.grid[action_y][action_x].type = self.char
        if self.char == O:
            if self.game.is_win(action_x, action_y):
                self.game.winner = self
                reward = self.win_reward
            elif not len(self.table.get_empty_cells()) > 0:
                reward = self.drow_reward
            else:
                reward = self.step_reward
            self.game.need_check_win = False
            self.update_q_table(state_hash, dirty_action_hash, reward, Q_max)
        elif self.char == X:
            if self.game.is_win(action_x, action_y):
                self.game.winner = self
                reward = self.lose_reward
                self.game.player_2.update_q_table(self.game.player_2.last_state_hash, self.game.player_2.last_action_hash, reward, self.game.player_2.last_Q_max)
            elif not len(self.table.get_empty_cells()) > 0:
                reward = self.drow_reward
                self.game.player_2.update_q_table(self.game.player_2.last_state_hash, self.game.player_2.last_action_hash, reward, self.game.player_2.last_Q_max)
        
        
    def update_q_table(self, state_hash, action_hash, reward, Q_max):
        td_target = reward + self.gamma * Q_max
        td_error = td_target - self.q_table[state_hash][action_hash]
        self.q_table[state_hash][action_hash] += self.alpha * td_error
        self.last_state_hash = state_hash
        self.last_action_hash = action_hash
        self.last_Q_max = Q_max
        
    def step(self, state):
        def get_coords(action_state):
            for y in range(self.table.height):
                for x in range(self.table.width):
                    if action_state[y][x] == '.':
                        return x, y
            
        action_state, Q_max, hash, dirty_action_hash = self.get_action(state)
        action_x, action_y = get_coords(action_state)
        return action_x, action_y, Q_max, hash, dirty_action_hash
            
    def get_action(self, state):
        hash, num = self.find_state_in_q_table(state)
        if not hash:
            hash = self.add_hash_to_q_table(state)
        if self.policy == 'greedy':
            dirty_action_hash = max(self.q_table[hash], key=self.q_table[hash].get)
        elif self.policy == 'egreedy':
            if np.random.random() < self.epsilon:
                dirty_action_hash = np.random.choice(list(self.q_table[hash].keys()))
            else:
                dirty_action_hash = max(self.q_table[hash], key=self.q_table[hash].get)
        Q_max = self.q_table[hash][dirty_action_hash]
        action_state = self.reverse_hash_and_get_state(dirty_action_hash, num)
        return action_state, Q_max, hash, dirty_action_hash
        
    def find_state_in_q_table(self, state):
        for num, similar_state in enumerate(self.similar_states_generator(state.copy())):
            hash = self.get_hash(similar_state)
            if hash in self.q_table:
                return hash, num
        return None, 0
         
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
                    actions[self.get_hash(state)] = BASE_Q
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
        with open(f'{self.name}_q_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def mirror_state_by_char(self, state):
        if self.char == X:
            return state
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
        
   
class Game:
    def __init__(self, state):
        self.table = Table(state)                                      #  RealPlayer(self.table, X)
        self.player_1 = QAgent(self.table, X, self, name='O')    #  QAgent(self.table, X, self, name='X')    PrimalAgent(self.table, X)
        self.player_2 = QAgent(self.table, O, self, name='O')  # PrimalAgent(self.table, O)  QAgent(self.table, O, self, name='O')
        self.players = [self.player_1, self.player_2]
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
        
        for player in self.players:
            if player.__class__ is QAgent:
                player.save_q_table()
        
    def cicle(self):
        while True:
            for player in self.players:
                self.move_counter += 1
                player.move()
                if self.need_check_win:
                    if self.is_win(player.last_cell.x, player.last_cell.y):
                        self.winner = player
                        break
                else:
                    self.need_check_win = True
                    if self.winner:
                        break
                if not self.table.get_empty_cells():
                    break
            else:
                continue
            break
        if self.winner:
            if self.winner is self.player_1:
                self.win_X_counter += 1
                if self.player_2.__class__ is QAgent:
                    self.player_2.update_q_table(self.player_2.last_state_hash, self.player_2.last_action_hash, 
                                                 self.player_2.lose_reward, self.player_2.last_Q_max)
            else:
                self.win_O_counter += 1
                if self.player_1.__class__ is QAgent:
                    self.player_1.update_q_table(self.player_1.last_state_hash, self.player_1.last_action_hash, 
                                                 self.player_1.lose_reward, self.player_1.last_Q_max)
        else:
            self.drow_counter += 1
            for player in self.players:
                if player.__class__ is QAgent:
                    player.update_q_table(player.last_state_hash, player.last_action_hash, 
                                                 player.drow_reward, player.last_Q_max)
            
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



app = Flask(__name__)

@app.route('/api', methods=['POST'])
def api_endpoint():
    try:
        # Получаем JSON из запроса
        state = np.array(request.get_json())
        game = Game(state)
        count_O = len(game.table.get_O_cells())
        count_X = len(game.table.get_X_cells())
        if count_O > count_X:
            return jsonify({"error": "Impossible state received. Too many 'O' cells"}), 500
        elif count_X - count_O > 1:
            return jsonify({"error": "Impossible state received. Too many 'X' cells"}), 500  
        game.player_2.move()
        state = game.table.get_state().tolist()
        if not game.winner and len(game.table.get_empty_cells()) > 0:
            game.player_1.move()
        game.player_2.save_q_table()

        # Отправляем JSON в ответ
        return jsonify(state)

    except Exception as e:
        # В случае ошибки возвращаем JSON с сообщением об ошибке
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Запускаем Flask приложение
    app.run(debug=True, host='0.0.0.0', port=5000)
    