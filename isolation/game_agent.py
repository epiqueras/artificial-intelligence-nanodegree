"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""


class SearchTimeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Player moves - opponent moves
    return float(len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player))))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Player moves - opponent moves
    player_moves = game.get_legal_moves(player)
    player_moves_len = len(player_moves)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    opponent_moves_len = len(opponent_moves)
    score = float(player_moves_len - opponent_moves_len)

    # Check for possibility of a division (simple check)
    for p_move in player_moves:
        for o_move in opponent_moves:
            if p_move[0] == o_move[0] and p_move[1] == o_move[1]:  # Equal move, not divided
                return score

    return float("-inf" if opponent_moves_len > player_moves_len else "inf")


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Player moves - opponent moves
    opponent = game.get_opponent(player)
    opponent_moves = game.get_legal_moves(opponent)
    o_location = game.get_player_location(opponent)
    score = float(len(game.get_legal_moves(player)) - len(opponent_moves))

    # Min distance to edges
    score -= min(game.width - o_location[0], o_location[0])
    score -= min(game.height - o_location[1], o_location[1])

    return score


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def minimax_helper(self, game, depth):
        """
        Recursive Minimax Helper
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = (-1, -1)  # Null move
        if depth == 0:  # We reached the end, return the heuristic score and the Null move
            return self.score(game, self), best_move
        moves = game.get_legal_moves()
        if not moves:  # No legal moves means game over, return the game utility and the Null move
            return game.utility(self), best_move

        maximizing = game.active_player == self
        best_val = float("-inf" if maximizing else "inf")
        for move in moves:  # Loop over moves
            # Search the next level.
            # Don't do anything with it's best move since we only care about the top level one
            next_val, _ = self.minimax_helper(
                game.forecast_move(move), depth - 1)
            # Update values if necessary
            best_val, best_move = (
                max((best_val, best_move), (next_val, move)) if maximizing
                else min((best_val, best_move), (next_val, move))
            )

        return best_val, best_move  # Return the best value and its move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # Needed to pass function signature/interface test
        return self.minimax_helper(game, depth)[1]

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        try:
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            # Return the Null move since no move was found due to the timeout
            return (-1, -1)


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def alphabeta_helper(self, game, depth, alpha, beta):
        """Recursive Alpha-Beta Helper"""
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = (-1, -1)  # Null move
        if depth == 0:  # We reached the end, return the heuristic score and the Null move
            return self.score(game, self), best_move
        moves = game.get_legal_moves()
        if not moves:  # No legal moves means game over, return the game utility and the Null move
            return game.utility(self), best_move

        maximizing = game.active_player == self
        best_val = float("-inf" if maximizing else "inf")
        for move in moves:  # Loop over moves
            # Search the next level.
            # Don't do anything with it's best move since we only care about the top level one
            next_val, _ = self.alphabeta_helper(
                game.forecast_move(move), depth - 1, alpha, beta)

            if maximizing and next_val > best_val:
                # New max, update values
                best_val = next_val
                best_move = move
                alpha = max(alpha, best_val)
            elif not maximizing and next_val < best_val:
                # New min, update values
                best_val = next_val
                best_move = move
                beta = min(beta, best_val)

            if beta <= alpha:  # Alpha-Beta Pruning
                return best_val, best_move

        return best_val, best_move  # Return the best value and its move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # Needed to pass function signature/interface test
        return self.alphabeta_helper(game, depth, alpha, beta)[1]

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize best_move so that this function always returns something.
        best_move = (-1, -1)

        i = 0
        while True:
            i += 1
            try:
                best_move = self.alphabeta(game, i)
            except SearchTimeout:
                break

        # Return the best move from the last completed search iteration
        return best_move
