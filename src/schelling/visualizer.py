from typing import List
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colors as col
import numpy as np
from .model import Schelling

class SchellingVisualizer:
    def __init__(self, model: Schelling):
        self.model = model
        self._setup_colors()
    
    def _setup_colors(self) -> None:
        """Setup color scheme using colorblind-friendly palette."""
        # Base colors from Okabe & Ito colorblind-friendly palette
        colors = ['#ffffff',  # Empty cells (white)
                 '#56b4e9',  # Race 1 (sky blue)
                 '#e69f00',  # Race 2 (orange)
                 '#009e73',  # Race 3 (bluish green)
                 '#cc79a7',  # Race 4 (reddish purple)
                 '#0072b2',  # Race 5 (blue)
                 '#d55e00',  # Race 6 (vermillion)
                 '#f0e442',  # Race 7 (yellow)
                 '#999999',  # Race 8 (grey)
                 '#44aa99',  # Race 9 (teal)
                 '#aa4499']  # Race 10 (purple)
        
        self.cmap = col.LinearSegmentedColormap.from_list(
            'schelling', 
            [col.hex2color(color) for color in colors[:self.model.config.n_races + 1]], 
            N=self.model.config.n_races + 1
        )
    
    def animate_simulation(self, n_steps: int, interval: int = 200) -> animation.ArtistAnimation:
        """Create an animation of the simulation running for n_steps.
        
        Args:
            n_steps: Number of simulation steps
            interval: Delay between frames in milliseconds (default: 200)
        
        Returns:
            Animation object that can be saved as GIF or displayed
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Schelling Segregation Model - Initial State")
        
        self.frames = []
        
        # Add initial state
        self.frames.append([plt.imshow(
            self.model.grid[1:-1, 1:-1], 
            cmap=self.cmap,
            animated=True
        )])
        
        step_count = 0
        while step_count < n_steps:
            moves = self.model.update()
            step_count += 1
            
            if moves > 0 or step_count >= n_steps:
                self.frames.append([plt.imshow(
                    self.model.grid[1:-1, 1:-1], 
                    cmap=self.cmap,
                    animated=True
                )])
            
            ax.set_title(f"Schelling Segregation Model - Step {step_count}")
            
            if moves == 0:
                break
        
        return animation.ArtistAnimation(
            fig, 
            self.frames,
            interval=interval, 
            blit=True,
            repeat=True
        )
    
    def plot_current_state(self) -> None:
        """Plot the current state of the model."""
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(self.model.grid[1:-1, 1:-1], cmap=self.cmap)
        plt.title("Schelling Segregation Model")
        plt.colorbar()
        plt.draw()
        return fig 