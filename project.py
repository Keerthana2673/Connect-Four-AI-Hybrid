"""
Complete Connect Four AI Project - Ultra Optimized Edition
----------------------------------------------------------
This module implements a Connect Four board using bitboards for maximum speed,
with the same heuristic evaluation, Minimax, MCTS, and other algorithms.

Key optimizations:
- Bitboard representation (64-bit integers) for ultra-fast operations
- Heuristic caching using LRU cache
- __slots__ in MCTSNode
- Optimized move generation and win detection
"""

import math
import random
import time
from collections import defaultdict
from functools import lru_cache

# Optional: PyTorch for DQN
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. DQN agent will be disabled.")


# 1. Bitboard Connect Four Board Class
class ConnectFourBoard:
    """
    Connect Four board using bitboards for maximum performance.
    Each player's pieces are stored as a 64-bit integer (bitboard).
    """
    
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.height = [0] * cols  # Number of pieces in each column
        self.mask = [0] * cols    # Bitmask for each column
        self.board_p1 = 0          # Bitboard for player 1
        self.board_p2 = 0          # Bitboard for player 2
        self.current_player = 1    # 1 = player 1, 2 = player 2
        self._initialize_masks()
    
    def _initialize_masks(self):
        """Initialize column masks for bitboard operations."""
        for col in range(self.cols):
            self.mask[col] = 0
            for row in range(self.rows):
                self.mask[col] |= 1 << (col * (self.rows + 1) + row)
    
    def _get_position(self, col, row):
        """Get the bit position for a given column and row."""
        return 1 << (col * (self.rows + 1) + row)
    
    def drop_piece(self, col, player):
        """Drop a piece for 'player' into column 'col'. Returns True if successful."""
        if self.is_valid_location(col):
            # Get the position for the next available row in this column
            pos = self._get_position(col, self.height[col])
            if player == 1:
                self.board_p1 |= pos
            else:
                self.board_p2 |= pos
            self.height[col] += 1
            self.current_player = 3 - player
            return True
        return False
    
    def is_valid_location(self, col):
        """Check if column 'col' is not full."""
        return self.height[col] < self.rows
    
    def get_valid_moves(self):
        """Return list of columns that are not full."""
        return [c for c in range(self.cols) if self.height[c] < self.rows]
    
    def winning_move(self, player):
        """
        Check if 'player' has a winning line using bitboard operations.
        This is extremely fast compared to the list-based version.
        """
        board = self.board_p1 if player == 1 else self.board_p2
        
        # Horizontal check
        m = board & (board >> (self.rows + 1))
        if m & (m >> (2 * (self.rows + 1))):
            return True
        
        # Vertical check
        m = board & (board >> 1)
        if m & (m >> 2):
            return True
        
        # Diagonal check (forward slash /)
        m = board & (board >> self.rows)
        if m & (m >> (2 * self.rows)):
            return True
        
        # Diagonal check (backward slash \)
        m = board & (board >> (self.rows + 2))
        if m & (m >> (2 * (self.rows + 2))):
            return True
        
        return False
    
    def is_full(self):
        """Return True if the board has no empty cells."""
        return all(h == self.rows for h in self.height)
    
    def copy(self):
        """Return a deep copy of the board."""
        new_board = ConnectFourBoard(self.rows, self.cols)
        new_board.height = self.height[:]
        new_board.board_p1 = self.board_p1
        new_board.board_p2 = self.board_p2
        new_board.current_player = self.current_player
        return new_board
    
    def reset(self):
        """Reset the board to empty."""
        self.height = [0] * self.cols
        self.board_p1 = 0
        self.board_p2 = 0
        self.current_player = 1
    
    def print_board(self):
        """Display the board (for debugging)."""
        print("\nCurrent board:")
        for row in range(self.rows - 1, -1, -1):
            row_str = "|"
            for col in range(self.cols):
                pos = self._get_position(col, row)
                if self.board_p1 & pos:
                    row_str += " X |"
                elif self.board_p2 & pos:
                    row_str += " O |"
                else:
                    row_str += "   |"
            print(row_str)
        print("  " + "   ".join(str(i + 1) for i in range(self.cols)) + "  ")
        print()
    
    def get_state_tensor(self):
        """Return a 2xrowsxcols tensor for neural network input."""
        state = torch.zeros((2, self.rows, self.cols), dtype=torch.float32)
        for col in range(self.cols):
            for row in range(self.rows):
                pos = self._get_position(col, row)
                if self.board_p1 & pos:
                    state[0, row, col] = 1.0
                elif self.board_p2 & pos:
                    state[1, row, col] = 1.0
        return state
    
    def get_board_array(self):
        """
        Convert bitboard to 2D list (for heuristic evaluation).
        Note: This is only used for the heuristic - the bitboard is faster for everything else.
        """
        board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for col in range(self.cols):
            for row in range(self.rows):
                pos = self._get_position(col, row)
                if self.board_p1 & pos:
                    board[row][col] = 1
                elif self.board_p2 & pos:
                    board[row][col] = 2
        return board


# 2. Heuristic Evaluation with Caching 
def _board_to_key(board, player):
    """Convert board and player to a hashable key for caching."""
    return (board.board_p1, board.board_p2, board.height_tuple(), player)

def _board_to_key_slow(board, player):
    """Fallback for backward compatibility."""
    return (tuple(tuple(row) for row in board.get_board_array()), player)

@lru_cache(maxsize=500000)
def _heuristic_cached(board_p1, board_p2, height_tuple, player, rows, cols):
    """
    Cached heuristic that supports dynamic board sizes.
    """

    board_array = [[0 for _ in range(cols)] for _ in range(rows)]

    for col in range(cols):
        for row in range(rows):
            pos = 1 << (col * (rows + 1) + row)
            if board_p1 & pos:
                board_array[row][col] = 1
            elif board_p2 & pos:
                board_array[row][col] = 2

    opponent = 3 - player
    total_score = 0

    def evaluate_window(window):
        score = 0
        if window.count(player) == 4:
            score += 100
        elif window.count(player) == 3 and window.count(0) == 1:
            score += 10
        elif window.count(player) == 2 and window.count(0) == 2:
            score += 2
        if window.count(opponent) == 3 and window.count(0) == 1:
            score -= 8
        return score

    # Center preference
    center_col = cols // 2
    center_count = sum(1 for r in range(rows) if board_array[r][center_col] == player)
    total_score += center_count * 3

    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            window = [board_array[r][c + i] for i in range(4)]
            total_score += evaluate_window(window)

    # Vertical
    for r in range(rows - 3):
        for c in range(cols):
            window = [board_array[r + i][c] for i in range(4)]
            total_score += evaluate_window(window)

    # Diagonal /
    for r in range(rows - 3):
        for c in range(cols - 3):
            window = [board_array[r + i][c + i] for i in range(4)]
            total_score += evaluate_window(window)

    # Diagonal \
    for r in range(3, rows):
        for c in range(cols - 3):
            window = [board_array[r - i][c + i] for i in range(4)]
            total_score += evaluate_window(window)

    return total_score

def heuristic(board, player):
    key = (board.board_p1, board.board_p2, tuple(board.height), player, board.rows, board.cols)
    return _heuristic_cached(*key)

def height_tuple(self):
    """Helper to convert height list to tuple for caching."""
    return tuple(self.height)

# Add the method to the board class
ConnectFourBoard.height_tuple = height_tuple


# 3. Minimax with Alpha-Beta Pruning 
def minimax(board, depth, alpha, beta, maximizing_player, player):
    """
    Minimax search with alpha-beta pruning.
    Returns (best_column, best_value) for the current player.
    """
    valid_moves = board.get_valid_moves()
    terminal = board.winning_move(1) or board.winning_move(2) or not valid_moves

    if depth == 0 or terminal:
        if terminal:
            if board.winning_move(player):
                return (None, 1000000)
            if board.winning_move(3 - player):
                return (None, -1000000)
            return (None, 0)
        return (None, heuristic(board, player))

    if maximizing_player:
        value = -float('inf')
        best_col = valid_moves[0]
        for col in valid_moves:
            new_board = board.copy()
            new_board.drop_piece(col, player)
            new_score = minimax(new_board, depth - 1, alpha, beta, False, player)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = float('inf')
        best_col = valid_moves[0]
        opponent = 3 - player
        for col in valid_moves:
            new_board = board.copy()
            new_board.drop_piece(col, opponent)
            new_score = minimax(new_board, depth - 1, alpha, beta, True, player)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value


def minimax_agent(board, player, depth=4):
    """Minimax agent with given search depth."""
    col, _ = minimax(board, depth, -float('inf'), float('inf'), True, player)
    return col


# 4. Monte Carlo Tree Search with __slots__ 
class MCTSNode:
    """Node for MCTS tree with __slots__ for memory efficiency."""
    __slots__ = ('board', 'parent', 'move', 'player', 'children', 'visits', 'wins', 'untried_moves')
    
    def __init__(self, board, parent=None, move=None, player=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.player = player
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = board.get_valid_moves()

    def uct_select_child(self, exploration_constant=1.414):
        """Select child with highest UCT value."""
        s = exploration_constant
        return max(
            self.children,
            key=lambda c: c.wins / c.visits +
            s * math.sqrt(math.log(self.visits) / c.visits)
        )

    def add_child(self, move, board, player):
        child = MCTSNode(board, parent=self, move=move, player=player)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result


def mcts(board, player, iterations=1000, heuristic_playouts=False):
    """
    Run MCTS from the current board state.
    If heuristic_playouts is True, use heuristic-biased simulations.
    """
    root = MCTSNode(board.copy(), player=3-player)

    for _ in range(iterations):
        node = root
        temp_board = board.copy()

        # Selection
        while node.untried_moves == [] and node.children != []:
            node = node.uct_select_child()
            temp_board.drop_piece(node.move, node.player)

        # Expansion
        if node.untried_moves:
            if heuristic_playouts:
                move = max(
                    node.untried_moves,
                    key=lambda m: heuristic_after_move(temp_board, m, 3 - node.player)
                )
            else:
                move = random.choice(node.untried_moves)
            next_player = 3 - node.player
            temp_board.drop_piece(move, next_player)
            node = node.add_child(move, temp_board, next_player)

        # Simulation
        sim_player = node.player
        while not (temp_board.winning_move(1) or temp_board.winning_move(2) or temp_board.is_full()):
            valid = temp_board.get_valid_moves()
            if not valid:
                break
            if heuristic_playouts:
                move = heuristic_playout_move(temp_board, sim_player)
            else:
                move = random.choice(valid)
            temp_board.drop_piece(move, sim_player)
            sim_player = 3 - sim_player

        # Determine winner
        if temp_board.winning_move(1):
            winner = 1
        elif temp_board.winning_move(2):
            winner = 2
        else:
            winner = 0

        # Backpropagation
        while node is not None:
            if winner == 0:
                result = 0.5
            elif winner == node.player:
                result = 1.0
            else:
                result = 0.0
            node.update(result)
            node = node.parent

    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move


def heuristic_after_move(board, col, player):
    """Heuristic value if 'player' drops a piece in column 'col'."""
    b_copy = board.copy()
    b_copy.drop_piece(col, player)
    return heuristic(b_copy, player)


def heuristic_playout_move(board, player):
    """Choose move with probability proportional to heuristic value."""
    valid = board.get_valid_moves()
    if not valid:
        return None
    scores = [heuristic_after_move(board, m, player) for m in valid]
    min_score = min(scores)
    if min_score < 0:
        scores = [s - min_score + 1e-6 for s in scores]
    total = sum(scores)
    if total < 1e-12:
        return random.choice(valid)
    probs = [s / total for s in scores]
    return random.choices(valid, weights=probs)[0]


def mcts_agent(board, player, iterations=200, heuristic_playouts=False):
    """MCTS agent wrapper."""
    return mcts(board, player, iterations, heuristic_playouts)


# 5. Baseline Agents
def random_agent(board, player):
    """Random move selection."""
    valid = board.get_valid_moves()
    return random.choice(valid) if valid else None


def heuristic_agent(board, player):
    """Greedy agent: always pick the move that maximizes the heuristic."""
    valid = board.get_valid_moves()
    if not valid:
        return None
    return max(valid, key=lambda m: heuristic_after_move(board, m, player))


# 6. Hybrid Agent with Logging
hybrid_stats = defaultdict(lambda: {"minimax_calls": 0, "mcts_calls": 0})

def hybrid_agent(board, player, mm_depth=4, mcts_iters=200, complexity_threshold=20, stats_key=None):
    """
    Use minimax if number of valid moves is below threshold,
    otherwise use MCTS (heuristic-biased).
    """
    valid = board.get_valid_moves()
    filled_cells = sum(board.height)
    total_cells = board.rows * board.cols
    complexity = filled_cells / total_cells

    use_minimax = complexity < 0.4
    if stats_key is not None:
        if use_minimax:
            hybrid_stats[stats_key]["minimax_calls"] += 1
        else:
            hybrid_stats[stats_key]["mcts_calls"] += 1
    if use_minimax:
        return minimax_agent(board, player, depth=mm_depth)
    else:
        return mcts(board, player, iterations=mcts_iters, heuristic_playouts=True)


# 7. Deep Q-Network (DQN) Agent 
if TORCH_AVAILABLE:
    class DQN(nn.Module):
        def __init__(self, rows=6, cols=7):
            super(DQN, self).__init__()
            # Input: 2 channels (Player 1 pieces, Player 2 pieces)
            self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * rows * cols, 256)
            self.fc2 = nn.Linear(256, cols)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    class ReplayBuffer:
        def __init__(self, capacity=10000):
            self.capacity = capacity
            self.buffer = []
            self.position = 0

        def push(self, state, action, reward, next_state, done):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            # Filter out None next_states (terminal states) and stack
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.stack(states)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.bool)
            
            # For terminal states, we use a zero tensor for next_state
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)
            non_final_next_states = torch.stack([s for s in next_states if s is not None]) if any(non_final_mask) else torch.empty(0)
            
            return states, actions, rewards, non_final_next_states, non_final_mask, dones

        def __len__(self):
            return len(self.buffer)

    class DQNAgent:
        def __init__(self, rows=6, cols=7, learning_rate=1e-4, gamma=0.95):
            self.rows, self.cols = rows, cols
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = DQN(rows, cols).to(self.device)
            self.target_net = DQN(rows, cols).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
            self.memory = ReplayBuffer(20000)
            
            self.gamma = gamma
            self.epsilon = 1.0
            self.epsilon_decay = 0.9995
            self.epsilon_min = 0.1
            self.batch_size = 64
            self.target_update = 200
            self.steps_done = 0

        def select_action(self, board, training=False):
            import numpy as np
            import torch

            state_tensor = board.get_state_tensor().unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy().flatten()

            valid_moves = board.get_valid_moves()

            for i in range(len(q_values)):
                if i not in valid_moves:
                    q_values[i] = -1e9

            best_action = int(np.argmax(q_values))

            if best_action not in valid_moves:
                return random.choice(valid_moves)

            return best_action
            
        def train_step(self):
            if len(self.memory) < self.batch_size:
                return

            states, actions, rewards, next_states, mask, dones = self.memory.sample(self.batch_size)
            states, actions, rewards = states.to(self.device), actions.to(self.device), rewards.to(self.device)

            # Compute Q(s_t, a)
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

            # Compute V(s_{t+1}) for all next states
            next_q_values = torch.zeros(self.batch_size).to(self.device)
            if next_states.nelement() > 0:
                next_q_values[mask] = self.target_net(next_states.to(self.device)).max(1)[0].detach()

            # Compute expected Q values
            expected_q_values = rewards + (self.gamma * next_q_values)

            loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.steps_done += 1
            if self.steps_done % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_dqn(episodes=500, rows=6, cols=7):
        agent = DQNAgent(rows, cols)
        board = ConnectFourBoard(rows, cols)
        
        print(f"Starting training on {agent.device}...")
        for ep in range(episodes):
            board.reset()
            state = board.get_state_tensor()
            done = False
            
            while not done:
                action = agent.select_action(board)
                # Store move result
                old_state = state
                board.drop_piece(action, board.current_player)
                
                reward = 0
                if board.winning_move(3 - board.current_player): # Previous player won
                    reward = 1.0
                    done = True
                elif board.is_full():
                    reward = 0.5
                    done = True
                
                state = board.get_state_tensor() if not done else None
                agent.memory.push(old_state, action, reward, state, done)
                agent.train_step()
            
            if (ep + 1) % 50 == 0:
                print(f"Episode {ep+1}/{episodes} | Epsilon: {agent.epsilon:.2f}")
        
        return agent

# Wrapper to use the agent in your tournament
def dqn_agent_wrapper(board, player, dqn_model_agent):
    return dqn_model_agent.select_action(board, training=False)

def timed_move(agent_func, board, player):
    start = time.time()
    move = agent_func(board, player)
    end = time.time()
    return move, (end - start)

# 8. Game Play and Tournament Functions 
def play_game(board, player1_func, player2_func, verbose=False):
    board.reset()
    current_player = 1

    time_taken = {1: 0, 2: 0}
    moves_count = {1: 0, 2: 0}

    while True:
        if current_player == 1:
            col, t = timed_move(player1_func, board, 1)
        else:
            col, t = timed_move(player2_func, board, 2)

        time_taken[current_player] += t
        moves_count[current_player] += 1

        if col is not None:
            assert 0 <= col < board.cols, f"Invalid column returned: {col}"

        if col is None or not board.is_valid_location(col):
            if verbose:
                print(f"Player {current_player} made invalid move!")
            return 3 - current_player, time_taken, moves_count

        board.drop_piece(col, current_player)

        if verbose:
            print(f"Player {current_player} -> col {col + 1}")
            board.print_board()

        if board.winning_move(current_player):
            return current_player, time_taken, moves_count

        if board.is_full():
            return 0, time_taken, moves_count

        current_player = 3 - current_player
        
                     

def tournament(board, player1_func, player2_func, games=100, verbose=False):
    wins1 = wins2 = draws = 0
    total_time = {1: 0, 2: 0}
    total_moves = {1: 0, 2: 0}

    for i in range(games):

        if i % 2 == 0:
            result, time_data, move_data = play_game(board, player1_func, player2_func, verbose)

            if result == 1:
                wins1 += 1
            elif result == 2:
                wins2 += 1
            else:
                draws += 1

        else:
            result, time_data, move_data = play_game(board, player2_func, player1_func, verbose)

            # Reverse interpretation
            if result == 2:
                wins1 += 1
            elif result == 1:
                wins2 += 1
            else:
                draws += 1

        for p in [1, 2]:
            total_time[p] += time_data[p]
            total_moves[p] += move_data[p]

        if verbose and (i + 1) % 10 == 0:
            print(f"After {i + 1} games: {wins1} - {wins2} - {draws}")

    avg_time_p1 = total_time[1] / max(1, total_moves[1])
    avg_time_p2 = total_time[2] / max(1, total_moves[2])

    return {
        "wins_p1": wins1,
        "wins_p2": wins2,
        "draws": draws,
        "avg_time_p1": avg_time_p1,
        "avg_time_p2": avg_time_p2
    }



# 9. Experiment Runner 
def run_experiments():
    print("\n=== Running Experiments ===")
    print("\nTraining DQN agent...")

    trained_dqn = train_dqn(episodes=500)
    print("DQN training completed.\n")

    board_sizes = [(6, 7), (7, 8), (8, 9)]
    num_games = 10  

    for rows, cols in board_sizes:
        print(f"\n===== Board Size: {rows} x {cols} =====")

        agents = {
            "Random": random_agent,
            "Heuristic": heuristic_agent,
            "Minimax (d=4)": lambda b, p: minimax_agent(b, p, depth=4),
            "MCTS (500)": lambda b, p: mcts_agent(b, p, iterations=500),
            "Hybrid": lambda b, p: hybrid_agent(b, p),
        }

        if (rows, cols) == (6, 7):
            agents["DQN (Trained)"] = lambda b, p: trained_dqn.select_action(b)

        agent_names = list(agents.keys())

        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):

                a1_name = agent_names[i]
                a2_name = agent_names[j]

                player1_func = agents[a1_name]
                player2_func = agents[a2_name]

                print(f"\n{a1_name} vs {a2_name}")

                result = tournament(
                    ConnectFourBoard(rows, cols),
                    player1_func,
                    player2_func,
                    games=num_games,
                    verbose=False
                )

                print(f"Results → "
                      f"{a1_name}: {result['wins_p1']} wins | "
                      f"{a2_name}: {result['wins_p2']} wins | "
                      f"Draws: {result['draws']}")

                print(f"Avg Time → "
                      f"{a1_name}: {result['avg_time_p1']:.6f}s | "
                      f"{a2_name}: {result['avg_time_p2']:.6f}s")



# 10. Main Entry Point
if __name__ == "__main__":
    run_experiments()

    # Optional: play a single verbose game between two chosen agents
    print("\n\n" + "=" * 70)
    print("SAMPLE GAME: Minimax (depth=4) vs MCTS(200,heur)")
    print("=" * 70)
    sample_board = ConnectFourBoard(6, 7)
    result, _, _ = play_game(
        sample_board,
        lambda b, p: minimax_agent(b, p, depth=4),
        lambda b, p: mcts_agent(b, p, iterations=200, heuristic_playouts=True),
        verbose=True
    )
    if result == 1:
        print("Player 1 (Minimax) wins!")
    elif result == 2:
        print("Player 2 (MCTS) wins!")
    else:
        print("It's a draw!")