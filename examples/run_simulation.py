from schelling.model import Schelling, SchellingConfig
from schelling.visualizer import SchellingVisualizer
import matplotlib.pyplot as plt

def main():
    # Create configuration
    config = SchellingConfig(
        grid_size=50,
        n_races=2,
        empty_ratio=0.1,
        similarity_threshold=0.6
    )
    
    # Initialize model and visualizer
    model = Schelling(config)
    viz = SchellingVisualizer(model)
    
    # Create animation with longer interval (500ms = 0.5s per frame)
    anim = viz.animate_simulation(n_steps=50, interval=500)
    
    # Save as GIF
    anim.save('schelling_simulation.gif', writer='pillow')
    
    plt.close()

if __name__ == "__main__":
    main() 