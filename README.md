# OptiViz
Walk along a loss landscape, watch from the sidelines, or ride along as your favourite optimisers battle it out for the best minimum with **OptiViz 1.0**, the latest update to OptiViz.
![Gradient descent with momentum minimising an egg-carton raised by a quadratic.](https://raw.githubusercontent.com/ronitkunk/optiviz/main/sgd_interactive.png)

Optiviz is a python package that enables effortless visualisation of *any* PyTorch optimiser on *any* differentiable function in one or two variables. OptiViz might find educational use in an introductory nonlinear optimisation or deep learning class.
![Vanilla gradient descent minimising a convex quadratic form.](https://raw.githubusercontent.com/ronitkunk/optiviz/main/sgd.png)

# Installation
To install OptiViz, please use:
```sh
pip install optiviz
```

# Usage
All functionality of OptiViz is exposed through the `optiviz.optimise` and `optiviz.optimise_interactive` functions.
```python
import torch
from optiviz import optimise, optimise_interactive
```
Any optimisation problem has an objective function. OptiViz works with differentiable, real-valued objective functions in one or two variables.
```math
f : \mathbb{R} \rightarrow \mathbb{R}
```
```math
g : \mathbb{R}^2 \rightarrow \mathbb{R}
```
In code, every input and output to the objective function must be a `torch.Tensor` of shape `(1,)`
```python
def f(x: torch.Tensor) -> torch.Tensor:
    """
    Example of an objective function in one variable.
    """
    return x ** 2
def g(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Example of an objective function in two variables.
    """
    return x ** 2 + y ** 2 + x.sin() * y.sin()
```
The `optiviz.optimise` function (please see docstring for complete usage) is used to visualise the optimisation sequence of a 1D or 2D objective function using a PyTorch optimiser.
```python
arg_g_min = optimise(
        g, # objective function
        (12.5, 12.5), # initial values of the parameters being adjusted
        plot_centre=(0, 0), # plot options
        plot_boundary=25,
        iters=100,
        optimiser=torch.optim.Adam, # PyTorch-compatible optimiser
        lr=5e-1 # any keyword arguments for the optimiser
    )
```
[NEW] The `optiviz.optimise_interactive` function (please see docstring for complete usage) provides interactive visualisation of 2D objective functions with multiple optimisers, navigation along the landscape, optimiser tracking, music and more.
```python
optimise_interactive(
        g, # objective function
        (12.5, 12.5), # initial values of the parameters being adjusted
        plot_centre=(0, 0), # plot options
        plot_boundary=25,
        iters=100,
        optimisers = [("Vanilla GD", lambda params: torch.optim.SGD(params, lr=0.1))], # PyTorch-compatible optimisers
        iter_delay: int = 100,
        gimbal_radius: float = 4.0, # advanced visualisation options
        gimbal_hover: float = 8.0,
    )
```

# Example programs
OptiViz 0.x/1.x:
```python
import torch
from optiviz import optimise

f = lambda x,y: ×**2+y**2

optimise(f, init_vector=(12.5, 12.5), optimiser=torch.optim.SGD, lr=le-2)
```

Optiviz 1.x:
```python
import torch
from optiviz import optimise_interactive

def egg_carton(x, y):
    return 0.05*(x**2+y**2)+2.5*(torch.sin(0.5*x)**2+torch.sin(0.5*y)**2)

optimise_interactive(fn=egg_carton, optimisers=[("SGD with momentum", lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.99)), ("Adam", lambda params: torch.optim.Adam(params, lr=0.5))])
```