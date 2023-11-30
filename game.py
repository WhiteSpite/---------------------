import numpy as np
import pickle

from agent import *


NUMBER_OF_GAMES = 10000
SHOW_GAME = 0
CREATE_PLOT_DATA = 0

if CREATE_PLOT_DATA:
    dataWin = []
    dataLoss = []
    dataDraw = []


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
                
    def get_action_coords(self, state):
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
        

class Game:
    def __init__(self):
        self.table = Table()
        self.player_X = QAgent(self.table, X, self, name='SimpleQ', epsilon=1, stop_learning=1)
        self.player_O = MeanQAgent(self.table, O, self, name='Mean', epsilon=0, stop_learning=1, to_cluster=0)
        self.players = [self.player_X, self.player_O]
        self.winner = None
        self.move_counter = 0
        self.drow_counter = 0
        self.win_X_counter = 0
        self.win_O_counter = 0
        self.need_check_win = True
        
    def play(self):
        if CREATE_PLOT_DATA:
            counter = 0
            global dataWin, dataLoss, dataDraw
        for _ in range(NUMBER_OF_GAMES):
            self.cicle()
            self.table.reset()
            self.winner = None
            self.players = self.players[::-1]
            self.move_counter = 0
            if CREATE_PLOT_DATA:
                counter += 1
                if counter % 100 == 0:
                    dataWin.append(self.win_O_counter)
                    dataLoss.append(self.win_X_counter)
                    dataDraw.append(self.drow_counter)
                    self.win_X_counter = 0
                    self.win_O_counter = 0
                    self.drow_counter = 0
            
        print(f'Draw {self.drow_counter}')
        print(f'WinX {self.win_X_counter}')
        print(f'WinO {self.win_O_counter}')
        
        for player in self.players:
            if QAgent in player.__class__.mro():
                player.save_q_table()
        
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
        for player in self.players:
            if QAgent in player.__class__.mro() and not player.stop_learning:
                if self.winner:
                    if self.winner is player:
                        result = 'win'
                    else:
                        result = 'loss'
                else:
                    result = 'draw'
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

if CREATE_PLOT_DATA:
    dataS = [dataWin, dataLoss, dataDraw]
    with open(f'plot_data.pkl', 'wb') as f:
        pickle.dump(dataS, f)
