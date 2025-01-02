from dataclasses import dataclass
from typing import List
import numpy as np
import numpy.typing as npt

@dataclass
class SchellingConfig:
    grid_size: int
    n_races: int
    empty_ratio: float
    similarity_threshold: float

class Schelling:
    def __init__(self, config: SchellingConfig):
        self.config = config
        self._initialize_grid()
        self._update_masks()
        
    def _initialize_grid(self) -> None:
        self.grid: npt.NDArray = np.random.choice(
            self.config.n_races + 1,
            size=(self.config.grid_size + 1) ** 2,
            p=[self.config.empty_ratio] + 
              [(1 - self.config.empty_ratio) / self.config.n_races] * self.config.n_races,
        ).reshape((self.config.grid_size + 1, self.config.grid_size + 1))
    
    def _update_masks(self) -> None:
        """Update the agent and empty cell masks."""
        grid_slice = self.grid[1:self.config.grid_size, 1:self.config.grid_size]
        self.agent_mask = grid_slice != 0
        self.empty_mask = ~self.agent_mask
    
    def _find_agents(self) -> tuple[np.ndarray, np.ndarray]:
        """Find all non-empty cells in the grid."""
        agent_rows, agent_cols = np.where(self.agent_mask)
        return agent_rows + 1, agent_cols + 1  # Account for border
    
    def _find_empty_cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Find all empty cells in the grid."""
        empty_rows, empty_cols = np.where(self.empty_mask)
        return empty_rows + 1, empty_cols + 1  # Account for border
    
    def _compute_all_satisfactions(self, agent_rows: np.ndarray, agent_cols: np.ndarray) -> np.ndarray:
        """Compute satisfaction ratios for all agents at once."""
        # Create 3x3 windows for each agent position
        windows = np.array([
            self.grid[r-1:r+2, c-1:c+2].flatten() 
            for r, c in zip(agent_rows, agent_cols)
        ])
        
        # Get agent types for each position
        agent_types = self.grid[agent_rows, agent_cols]
        
        # Remove center cells (the agents themselves)
        neighborhoods = np.column_stack([windows[:, :4], windows[:, 5:]])
        
        # Count non-empty and similar neighbors
        non_empty = (neighborhoods != 0).sum(axis=1)
        similar = (neighborhoods == agent_types[:, np.newaxis]).sum(axis=1)
        
        # Handle division by zero more explicitly
        satisfaction = np.zeros_like(non_empty, dtype=float)
        mask = non_empty > 0
        satisfaction[mask] = similar[mask] / non_empty[mask]
        
        return satisfaction
    
    def _find_unsatisfied_agents(self, agent_rows: np.ndarray, agent_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find all unsatisfied agents."""
        satisfactions = self._compute_all_satisfactions(agent_rows, agent_cols)
        unsatisfied = satisfactions < self.config.similarity_threshold
        return agent_rows[unsatisfied], agent_cols[unsatisfied]
    
    def _move_agents(self, from_rows: np.ndarray, from_cols: np.ndarray, 
                    to_rows: np.ndarray, to_cols: np.ndarray) -> None:
        """Move agents from one set of positions to another."""
        moving_types = self.grid[from_rows, from_cols]
        self.grid[to_rows, to_cols] = moving_types
        self.grid[from_rows, from_cols] = 0
        self._update_masks()  # Update masks after moving agents
    
    def update(self) -> int:
        """Perform one step of the simulation."""
        # Find all agents and empty cells
        agent_rows, agent_cols = self._find_agents()
        empty_rows, empty_cols = self._find_empty_cells()
        
        if len(empty_rows) == 0:
            return 0
        
        # Find unsatisfied agents
        unsatisfied_rows, unsatisfied_cols = self._find_unsatisfied_agents(agent_rows, agent_cols)
        
        if len(unsatisfied_rows) == 0:
            return 0
        
        # Randomly shuffle unsatisfied agents to give equal opportunity
        shuffle_idx = np.random.permutation(len(unsatisfied_rows))
        unsatisfied_rows = unsatisfied_rows[shuffle_idx]
        unsatisfied_cols = unsatisfied_cols[shuffle_idx]
        
        # Determine how many moves we can make
        n_moves = min(len(unsatisfied_rows), len(empty_rows))
        
        # Randomly select empty cells for the moves
        empty_indices = np.random.choice(len(empty_rows), size=n_moves, replace=False)
        to_rows = empty_rows[empty_indices]
        to_cols = empty_cols[empty_indices]
        
        # Move the agents
        self._move_agents(
            unsatisfied_rows[:n_moves], 
            unsatisfied_cols[:n_moves],
            to_rows, 
            to_cols
        )
        
        return n_moves
    
    def run_simulation(self, max_steps: int = 100, min_moves: int = 0) -> List[int]:
        """
        Run the simulation until either max_steps is reached or 
        number of moves falls below min_moves.
        Returns list of moves made in each step.
        """
        moves_history = []
        for _ in range(max_steps):
            moves = self.update()
            moves_history.append(moves)
            if moves <= min_moves:
                break
        return moves_history 