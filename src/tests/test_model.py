import pytest
import numpy as np
from schelling.model import Schelling, SchellingConfig

@pytest.fixture
def basic_config():
    return SchellingConfig(
        grid_size=10,
        n_races=2,
        empty_ratio=0.2,
        similarity_threshold=0.6
    )

def test_initialization(basic_config):
    model = Schelling(basic_config)
    assert model.grid.shape == (11, 11)  # +1 for borders
    assert set(np.unique(model.grid)) <= set(range(basic_config.n_races + 1))

def test_simulation_runs(basic_config):
    model = Schelling(basic_config)
    moves = model.run_simulation(max_steps=10)
    assert len(moves) <= 10
    assert all(isinstance(m, int) for m in moves) 

def test_update_makes_changes(basic_config):
    model = Schelling(basic_config)
    initial_state = model.grid.copy()
    moves = model.update()
    assert moves >= 0
    if moves > 0:
        assert not np.array_equal(initial_state, model.grid)

def test_convergence(basic_config):
    """Test that simulation eventually converges (moves become 0 or stabilize)"""
    model = Schelling(basic_config)
    # Set a very low similarity threshold to force convergence
    model.config.similarity_threshold = 0.1  # Agents will be easily satisfied
    moves = model.run_simulation(max_steps=1000, min_moves=0)
    
    # Check either for complete convergence or stability
    last_moves = moves[-10:]  # Look at last 10 moves
    assert len(set(last_moves)) <= 2  # Should stabilize to at most 2 different values
    assert min(last_moves) <= 1  # Should have very few moves at end

@pytest.mark.parametrize("n_races", [2, 3, 4])
def test_multiple_races(basic_config, n_races):
    """Test model works with different numbers of races"""
    config = SchellingConfig(
        grid_size=basic_config.grid_size,
        n_races=n_races,
        empty_ratio=basic_config.empty_ratio,
        similarity_threshold=basic_config.similarity_threshold
    )
    model = Schelling(config)
    unique_values = set(np.unique(model.grid))
    assert len(unique_values) <= n_races + 1  # +1 for empty cells
    assert max(unique_values) <= n_races
    assert min(unique_values) >= 0 

def test_mask_initialization(basic_config):
    """Test that masks are properly initialized."""
    model = Schelling(basic_config)
    assert hasattr(model, 'agent_mask')
    assert hasattr(model, 'empty_mask')
    assert model.agent_mask.shape == (basic_config.grid_size-1, basic_config.grid_size-1)
    assert model.empty_mask.shape == (basic_config.grid_size-1, basic_config.grid_size-1)
    assert np.all(model.empty_mask == ~model.agent_mask)

def test_masks_update_after_move(basic_config):
    """Test that masks are updated after agent moves."""
    model = Schelling(basic_config)
    initial_agent_mask = model.agent_mask.copy()
    initial_empty_mask = model.empty_mask.copy()
    
    # Force a move by setting up a specific grid state
    model.grid = np.zeros((basic_config.grid_size + 1, basic_config.grid_size + 1))
    model.grid[1, 1] = 1  # Add an agent
    model._update_masks()
    
    # Move the agent
    model._move_agents(
        np.array([1]), np.array([1]),  # from
        np.array([2]), np.array([2])   # to
    )
    
    # Check that masks were updated
    assert not np.array_equal(model.agent_mask, initial_agent_mask)
    assert not np.array_equal(model.empty_mask, initial_empty_mask)
    # Subtract 1 from grid positions to get mask positions (due to border)
    assert model.agent_mask[0, 0] == False  # Old position (1,1) in grid is (0,0) in mask
    assert model.agent_mask[1, 1] == True   # New position (2,2) in grid is (1,1) in mask

def test_find_agents_uses_mask(basic_config):
    """Test that _find_agents uses the mask."""
    model = Schelling(basic_config)
    
    # Set up a known state
    model.grid = np.zeros((basic_config.grid_size + 1, basic_config.grid_size + 1))
    model.grid[1:3, 1:3] = 1
    model._update_masks()
    
    agent_rows, agent_cols = model._find_agents()
    assert len(agent_rows) == 4  # Should find all agents
    assert np.all(model.agent_mask[0:2, 0:2])  # Check mask matches grid

def test_find_empty_cells_uses_mask(basic_config):
    """Test that _find_empty_cells uses the mask."""
    model = Schelling(basic_config)
    
    # Set up a known state
    model.grid = np.ones((basic_config.grid_size + 1, basic_config.grid_size + 1))
    model.grid[1:3, 1:3] = 0
    model._update_masks()
    
    empty_rows, empty_cols = model._find_empty_cells()
    assert len(empty_rows) == 4  # Should find all empty cells
    assert np.all(model.empty_mask[0:2, 0:2])  # Check mask matches grid

def test_masks_complement(basic_config):
    """Test that agent and empty masks are always complementary."""
    model = Schelling(basic_config)
    
    # Check after initialization
    assert np.all(model.agent_mask == ~model.empty_mask)
    
    # Check after update
    model.update()
    assert np.all(model.agent_mask == ~model.empty_mask) 

def test_compute_all_satisfactions(basic_config):
    """Test vectorized satisfaction computation."""
    model = Schelling(basic_config)
    
    # Set up a known grid state
    model.grid = np.zeros((basic_config.grid_size + 1, basic_config.grid_size + 1))
    model.grid[1:4, 1:4] = np.array([
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 2]
    ])
    model._update_masks()
    
    # Get agent positions
    agent_rows, agent_cols = model._find_agents()
    satisfactions = model._compute_all_satisfactions(agent_rows, agent_cols)
    
    # Check center agent's satisfaction
    center_idx = np.where((agent_rows == 2) & (agent_cols == 2))[0][0]
    assert satisfactions[center_idx] == 0.625  # 5 out of 8 neighbors are similar 

def test_update_with_no_empty_cells(basic_config):
    """Test that update returns 0 when there are no empty cells."""
    model = Schelling(basic_config)
    
    # Fill the grid completely with agents (no empty cells)
    model.grid = np.ones((basic_config.grid_size + 1, basic_config.grid_size + 1))
    model._update_masks()
    
    # Try to update
    moves = model.update()
    assert moves == 0 

def test_random_movement():
    """Test that unsatisfied agents move to random empty cells."""
    config = SchellingConfig(
        grid_size=5,
        n_races=2,
        empty_ratio=0.2,
        similarity_threshold=0.9  # High threshold to ensure agent is unsatisfied
    )
    model = Schelling(config)
    
    # Set up a grid with one unsatisfied agent and multiple empty cells
    model.grid = np.zeros((6, 6))
    model.grid[2, 2] = 1  # Center agent
    model.grid[1, 2] = 2  # Different neighbor to make agent unsatisfied
    model._update_masks()
    
    # Run multiple updates to check randomness
    empty_positions = set()
    for _ in range(10):
        model.grid = np.zeros((6, 6))
        model.grid[2, 2] = 1
        model.grid[1, 2] = 2
        model._update_masks()
        
        model.update()
        
        # Find where the agent moved
        new_pos = tuple(map(tuple, np.where(model.grid == 1)))[0]
        empty_positions.add(new_pos)
    
    # Agent should have moved to different positions
    assert len(empty_positions) > 1 