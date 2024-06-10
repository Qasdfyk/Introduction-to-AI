from minmax import MinMaxAlphaBeta
from dots_and_boxes import DotsAndBoxes
import random
import seaborn as sns
import matplotlib.pyplot as plt


def dots_and_boxes_2_computers(game_size, depth1, depth2):
    game = DotsAndBoxes(game_size)
    while not game.state.is_finished():
       # print(game.state.__str__())
        moves = game.state.get_moves()
        best_move = []

        if game.state.get_current_player().char == "1": #Max
            best_value = float("-inf")
            for move in moves:
                next_state = game.state.make_move(move)
                value = MinMaxAlphaBeta.solve(next_state, depth1, game.state.get_current_player().char)
              #  print(value)
                if value > best_value:
                    best_value = value
                    best_move.clear()
                    best_move.append(move)
                elif value == best_value:
                    best_move.append(move)

        elif game.state.get_current_player().char == "2": #Min
            best_value = float("inf")
            for move in moves:
                next_state = game.state.make_move(move)
                value = MinMaxAlphaBeta.solve(next_state, depth2, game.state.get_current_player().char)
               # print(value)
                if value < best_value:
                    best_value = value
                    best_move.clear()
                    best_move.append(move)
                elif value == best_value:
                    best_move.append(move)
      # print([(move.connection, move.loc) for move in best_move])
        game.state = game.state.make_move(random.choice(best_move))
        best_move.clear()
    if game.state.get_current_player().char == "1":
        return [game.state.get_scores()[game.state._other_player], game.state.get_scores()[game.state.get_current_player()]]
    else:
        return [game.state.get_scores()[game.state.get_current_player()], game.state.get_scores()[game.state._other_player]]

def random_game(size):
    game = DotsAndBoxes(size)
    while not game.state.is_finished():
        moves = game.state.get_moves()
        game.state = game.state.make_move(random.choice(moves))
    return [game.state.get_scores()[game.first_player], game.state.get_scores()[game.second_player]]

def win_percentages(size, max_depth, tries):
    data = []
    for i in range(max_depth+1): # depth 1
        row = []
        for j in range(max_depth+1): # depth 2
            print(i,j)
            win_percentage_of_1_in_try = 0
            for _ in range(tries):
                x = dots_and_boxes_2_computers(size, i, j)
                if x[0] > x[1]:
                    win_percentage_of_1_in_try += 100
                elif x[0] == x[1]:
                    win_percentage_of_1_in_try += 50
            row.append(win_percentage_of_1_in_try/tries)
        data.append(row)
    s = sns.heatmap(data, annot=True, fmt=".1f")
    s.set(xlabel='depth of second computer', ylabel='depth of first computer')
    plt.title("Win percentages of first computer")
    plt.show()

def diff_in_depths(max_depth):
    wins = []
    for i in range(max_depth):
        temp=0
        for _ in range(10):
            result = dots_and_boxes_2_computers(3, 1, i+1)
            if result[1] > result[0]:
                temp+=10
        wins.append(temp)
    return wins

def test_win_percentages_random_moves(size, tries):
    win_of_1 = 0
    for _ in range(tries):
        x = random_game(size)
        if x[0] > x[1]:
            win_of_1 += 100
        elif x[0] == x[1]:
            win_of_1 += 50
    return win_of_1/tries


if __name__ == "__main__":
    win_percentages(3, 4, 100)
    #print(dots_and_boxes_2_computers(3, 1, 2))
    #print(diff_in_depths(4))
    #print(test_win_percentages_random_moves(5, 1000)) 
    #print(test_win_percentages_random_moves(4, 10000)) 
    #print(test_win_percentages_random_moves(3, 10000)) 
