# Schelling Segregation Model

A Python implementation of Thomas Schelling's segregation model, demonstrating how individual preferences can lead to emergent segregation patterns.

## Overview

The Schelling segregation model shows how a small preference for similar neighbors can lead to large-scale segregation patterns. In this implementation:

- Agents of different types (races) are placed randomly on a grid
- Some cells are left empty to allow movement
- Agents are "satisfied" if a certain percentage of their neighbors are similar
- Unsatisfied agents randomly move to empty cells
- The process continues until all agents are satisfied or no more moves are possible

## Installation

Clone the repository
```bash
git clone https://github.com/yourusername/schelling.git
cd schelling
```

Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation with default parameters:
```bash
python examples/run_simulation.py
```

This will create a GIF animation showing the evolution of the segregation patterns.

### Configuration

The model can be customized with different parameters:

```python
from schelling.model import Schelling, SchellingConfig
from schelling.visualizer import SchellingVisualizer

config = SchellingConfig(
    grid_size=50,        # Size of the grid (50x50)
    n_races=2,           # Number of different agent types
    empty_ratio=0.1,     # Proportion of cells left empty
    similarity_threshold=0.6  # Minimum ratio of similar neighbors for satisfaction
)

model = Schelling(config)
viz = SchellingVisualizer(model)
```

## Testing

Run the test suite:

```bash
pytest
```

