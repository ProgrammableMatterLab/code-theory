
block is a library designed to study and visualize encodings and their interactions under rotation and translation.

# Key Components

1. **Block Class**: Encapsulates tensor data with polarity information for calculating attraction between blocks.
2. **Plotting Functions**: Include `plot_faces` for visualizing tensor faces with polarities.
3. **Utility Functions**: Convert tensors to points using `tensor_to_points`.

# Installation
To install the block library, follow these steps:

## Pre-requisites:
- Python 3.8 or higher
- pip
## Installation via GitHub:
1. Clone the repository:
```bash
git clone https://github.com/your-username/block.git
```
2. Navigate into the cloned repository:
```bash
cd block
```
3. Install the package in editable mode:
```bash
pip3 install -e .
```
## Requirements:
Ensure you have the necessary dependencies installed. You can install them using:
```bash
pip install torch scipy numpy matplotlib
```
Or, if you prefer to manage dependencies via a requirements.txt file, you can create one with the following content:
```text
torch
scipy
numpy
matplotlib
```
Then, install the dependencies using:
```bash
pip3 install -r requirements.txt
```
## Optional: Conda Environment Setup
For those who prefer using Conda, you can create a dedicated environment:
```bash
conda create --name block-env python=3.8
conda activate block-env
pip install git+https://github.com/your-username/block.git
```
Make sure to replace your-username with your actual GitHub username and adjust the repository URL accordingly.
### Notes
- Ensure you have Git installed if you choose to clone the repository.
- If you encounter issues with dependencies, consider using a virtual environment to manage them more effectively.

# Testing

to run all tests, run
```bash
python3 -m unittest discover -s block/tests -p 'test_*.py'
```

to run individual test, run
```bash
python3 -m unittest discover -s block/tests -p 'test_{test name here}.py'
```

# API Reference

## Block Class

The `Block` class is the core component of the library, representing a block structure with associated tensor data and polarities.

#### Constructor

```python
Block(tensor: torch.Tensor)
```

Initializes a Block object with a 2D tensor representing the block's structure.

#### Class Methods

```python
@classmethod
from_block(cls, other: Block) -> Block
```

Creates a new Block instance by cloning another Block.

#### Static Methods

```python
@staticmethod
calculate_attraction(block1: Block, block2: Block) -> Tuple[torch.Tensor, float]
```

Calculates the attraction between two blocks, returning a tensor of attraction forces and the sum of forces normalized by the number of elements in block1.

#### Instance Methods

```python
rotated(self, theta: float, mode: str = 'd') -> Block
```

Returns a new Block with points rotated by the given angle theta.

```python
rotate(self, theta: float, mode: str = 'd') -> None
```

Rotates the points of the current Block by the given angle theta in-place.

## Plotting Functions

```python
plot_faces(blocks: Union[Block, List[Block]], colors: List[str], zoom_factor: float = 2, alpha: float = 0.5) -> None
```

Visualizes one or more Blocks, displaying their points as circles with polarities indicated by 'N' or 'S' markers.

![Visualizing Blocks](images/plot_faces.png)

## Usage Example

check out examples/

## Testing
    $ python -m unittest discover -s block/tests

### TODO
- [ ] For plots showing overlapping N & S pixels: show a distinct color for attraction, another color for repulsion
- [ ] Don't normalize - we care about overlapping areas to compare different resolutions
- [ ] Animation of the rotating surfaces alongside a graph
- [ ] Earnshaw's theorem
- [ ] non-linear programming for designing N/S placement to shape a potential well

## References

1. **SGDAT: An Optimization Method for Binary Neural Networks**
    - https://arxiv.org/pdf/2302.11062
2. **DPCD: Discrete Principal Coordinate Descent for Binary Variable Problems** by Huan Xiong
    - https://econ.la.psu.edu/wp-content/uploads/sites/5/2022/01/GenRanCorr.pdf
3. **Determinant Optimization on Binary Matrices**
    - https://www.researchgate.net/publication/216813262_Determinant_Optimization_on_Binary_Matrices
4. **The Hadamard decomposition problem**
    - https://www.researchgate.net/publication/380756108_The_Hadamard_decomposition_problem
5. https://openreview.net/forum?id=rvhu4V7yrX
6. https://dspace.mit.edu/bitstream/handle/1721.1/108443/Soljacic_Binary matrices.pdf?sequence=1
7. https://arxiv.org/pdf/2110.02560
8. https://www.researchgate.net/publication/220240951_New_Matrices_with_Good_Auto_and_Cross-Correlation
9. https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-017-0455-2
10. https://pubmed.ncbi.nlm.nih.gov/33449928/
11. https://ocw.mit.edu/courses/6-972-algebraic-techniques-and-semidefinite-optimization-spring-2006/813f1063132bcd1abe4283d7c9f75816_lecture_03.pdf
12. https://users.cs.duke.edu/~reif/paper/urmi/magneticAssembly/magneticAssembly.pdf
13. https://www.researchgate.net/publication/221344528_Three_Dimensional_Stochastic_Reconfiguration_of_Modular_Robots
14. https://journals.aps.org/prx/pdf/10.1103/PhysRevX.14.021004
15. https://www.nature.com/articles/s41467-022-32892-y
16. https://ijmttjournal.org/public/assets/volume-59/number-4/IJMTT-V59P532.pdf
