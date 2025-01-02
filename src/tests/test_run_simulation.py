import pytest
import os
import matplotlib.pyplot as plt
from schelling.model import Schelling, SchellingConfig
from schelling.visualizer import SchellingVisualizer
from matplotlib import animation

def test_simulation_creates_video():
    """Test that simulation creates a video animation"""
    config = SchellingConfig(
        grid_size=10,  # Smaller grid for testing
        n_races=2,
        empty_ratio=0.2,
        similarity_threshold=0.6
    )
    
    model = Schelling(config)
    viz = SchellingVisualizer(model)
    anim = viz.animate_simulation(n_steps=5)  # Fewer steps
    
    assert isinstance(anim, animation.ArtistAnimation)
    assert len(viz.frames) > 0
    plt.close()  # Clean up

def test_simulation_parameters():
    """Test that simulation uses correct parameters"""
    config = SchellingConfig(
        grid_size=10,  # Smaller grid
        n_races=2,
        empty_ratio=0.2,
        similarity_threshold=0.6
    )
    
    model = Schelling(config)
    viz = SchellingVisualizer(model)
    anim = viz.animate_simulation(n_steps=5, interval=200)
    
    assert anim._interval == 200
    assert len(viz.frames) > 0
    plt.close()  # Clean up 