import pytest
from schelling.model import Schelling, SchellingConfig
from schelling.visualizer import SchellingVisualizer
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
from matplotlib.image import AxesImage

@pytest.fixture
def model():
    config = SchellingConfig(
        grid_size=10,
        n_races=2,
        empty_ratio=0.2,
        similarity_threshold=0.6
    )
    return Schelling(config)

def test_visualizer_initialization(model):
    viz = SchellingVisualizer(model)
    assert hasattr(viz, 'cmap')
    assert hasattr(viz, 'model')

def test_animation_creation(model):
    viz = SchellingVisualizer(model)
    anim = viz.animate_simulation(n_steps=5)
    assert anim is not None

def test_plot_current_state(model):
    viz = SchellingVisualizer(model)
    # This just tests that the method runs without error
    viz.plot_current_state() 

def test_animation_stops_on_convergence(model):
    """Test that animation stops when no more moves are possible"""
    viz = SchellingVisualizer(model)
    
    size = model.config.grid_size + 1
    stable_grid = np.zeros((size, size))
    inner_grid = np.ones((size-2, size-2))
    stable_grid[1:-1, 1:-1] = inner_grid
    
    model.grid = stable_grid
    anim = viz.animate_simulation(n_steps=50, interval=100)
    assert len(viz.frames) == 1
    plt.close()

def test_animation_parameters(model):
    """Test that animation parameters are correctly set"""
    viz = SchellingVisualizer(model)
    interval = 200
    anim = viz.animate_simulation(n_steps=10, interval=interval)
    
    assert anim._interval == interval
    assert anim._repeat == True
    assert anim._blit == True
    plt.close()

@pytest.mark.parametrize("n_races", [2, 3, 4])
def test_colormap_matches_races(model, n_races):
    """Test that colormap has correct number of colors for n_races"""
    model.config.n_races = n_races
    viz = SchellingVisualizer(model)
    
    # Colormap should have n_races + 1 colors (including empty cell color)
    assert viz.cmap.N == n_races + 1

def test_plot_current_state_with_colorbar(model):
    """Test that plot_current_state includes a colorbar"""
    viz = SchellingVisualizer(model)
    fig = viz.plot_current_state()  # Get the figure directly
    plt.pause(0.1)  # Give it a moment to render
    assert len(fig.axes) > 1, "Colorbar not found in figure"
    plt.close(fig)

def test_animation_title_updates(model):
    """Test that animation title updates with step count"""
    viz = SchellingVisualizer(model)
    n_steps = 5
    
    anim = viz.animate_simulation(n_steps=n_steps, interval=100)
    
    # Check the title of the last frame
    ax = plt.gca()
    assert ax.get_title().endswith(f"Step {n_steps}")
    
    plt.close() 

def test_animation_includes_initial_state(model):
    """Test that animation includes the initial state as first frame"""
    viz = SchellingVisualizer(model)
    initial_state = model.grid.copy()
    anim = viz.animate_simulation(n_steps=5, interval=100)
    
    first_frame = viz.frames[0][0]
    first_frame_data = first_frame.get_array()
    
    assert np.array_equal(first_frame_data, initial_state[1:-1, 1:-1])
    assert len(viz.frames) > 1
    plt.close() 

@pytest.mark.parametrize("n_races", [2, 3, 4])
def test_patterns_match_races(model, n_races):
    """Test that correct number of colors are used for n_races."""
    model.config.n_races = n_races
    viz = SchellingVisualizer(model)
    
    # Plot current state
    fig = viz.plot_current_state()
    
    # Check that colormap has correct number of colors
    assert viz.cmap.N == n_races + 1  # +1 for empty cells
    plt.close(fig)

def test_colorblind_friendly_palette(model):
    """Test that colors are from the Okabe-Ito colorblind-friendly palette."""
    viz = SchellingVisualizer(model)
    expected_colors = ['#ffffff', '#56b4e9', '#e69f00']  # First few colors
    
    # Convert colormap colors to hex
    cmap_colors = [mcolors.rgb2hex(viz.cmap(i)[:3]) for i in range(model.config.n_races + 1)]
    
    # Check that the first few colors match expected
    assert all(c1.lower() == c2.lower() for c1, c2 in zip(cmap_colors, expected_colors)) 
  
def test_animation_gif_compatibility(model, tmp_path):
    """Test that animation can be saved as GIF."""
    viz = SchellingVisualizer(model)
    anim = viz.animate_simulation(n_steps=5, interval=200)
    
    # Save as GIF to temporary directory
    gif_path = tmp_path / "test.gif"
    anim.save(str(gif_path), writer='pillow')
    
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    plt.close() 
  
def test_animation_frame_generation(model):
    """Test that frames are generated correctly."""
    viz = SchellingVisualizer(model)
    n_steps = 10
    
    initial_state = model.grid.copy()
    anim = viz.animate_simulation(n_steps=n_steps)
    
    # Should have at least initial frame plus one more
    assert len(viz.frames) >= 2
    assert not np.array_equal(initial_state, model.grid)
    plt.close() 
  