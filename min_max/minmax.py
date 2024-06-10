
class MinMaxAlphaBeta:
    @staticmethod
    def solve(state, depth, move_max, alpha=float("-inf"), beta=float("inf")):
        if depth == 0 or state.is_finished():
            if state.get_current_player().char == "1":
                return state.get_scores()[state.get_current_player()] - state.get_scores()[state._other_player]
            else:
                return state.get_scores()[state._other_player] - state.get_scores()[state.get_current_player()]
        u = state.get_moves()
        if move_max == "1":
            for move in u:
                next_state = state.make_move(move)
                alpha = max(alpha, MinMaxAlphaBeta.solve(next_state, depth-1, next_state.get_current_player().char, alpha, beta))
                if alpha >= beta:
                    return alpha
            return alpha
        else:
            for move in u:
                next_state = state.make_move(move)
                beta = min(beta, MinMaxAlphaBeta.solve(next_state, depth-1, next_state.get_current_player().char, alpha, beta))
                if alpha >= beta:
                    return beta
            return beta
