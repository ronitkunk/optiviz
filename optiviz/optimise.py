import torch
import inspect
import threading
import time
import webbrowser
from typing import Type, Callable, List, Tuple, Optional
import matplotlib.pyplot as plt
import os
from flask import Flask, jsonify, send_from_directory


from .visualise import plot_objective_1D, plot_point_1D, plot_objective_2D, plot_point_2D

def optimise(fn, init_vector: tuple[float], plot_centre: tuple[float]=None, plot_boundary: float=25, iters: int=1000, optimiser: Type[torch.optim.Optimizer]=torch.optim.Adam, **kwargs) -> tuple[float]:
    """
    Visualises the minimisation sequence of the given differentiable function `fn` using the given optimiser.
    Arguments:
        `fn` : The differentiable function to be minimised. Must take exactly 1 or 2 non-default arguments. Return value and each argument must be a `torch.Tensor` with shape (1,).
        `init_vector`: A tuple of the same dimension as the number of arguments of `fn`, specifying the initial values of the function parameters.
        `plot_centre`: A tuple of the same dimension as the number of arguments of `fn`, specifying the centre point of the plot (in parameter space); defaults to 0 in all dimensions.
        `plot_boundary`: Length of the the plot boundary in all dimensions in the parameter space.
        `iters`: Number of optimiser iterations; defaults to 1000.
        `optimiser`: Optimisation algorithm to use. Must be a `torch.optim.Optimizer` subclass (not instance); defaults to Adam (https://arxiv.org/abs/1412.6980).
        `**kwargs`: any keyword arguments for the optimiser; e.g. lr.

    Returns:
        Depending on fn, a 1-tuple or 2-tuple of the estimated optimal parameters.
    """
    sig = inspect.signature(fn)
    input_dim = sum(p.default == inspect._empty for p in sig.parameters.values())

    if plot_centre is None:
        plot_centre = tuple([0 for _ in range(input_dim)])

    assert input_dim==1 or input_dim==2, f"'fn' must take either 1 or 2 non-default arguments (received {input_dim})."
    assert input_dim==len(init_vector), f"Number of non-default arguments of 'fn' ({input_dim}) does not match length of 'init_vector' ({len(init_vector)})."
    assert input_dim==len(plot_centre), f"Number of non-default arguments of 'fn' ({input_dim}) does not match length of 'plot_centre' ({len(plot_centre)})."

    x = tuple([torch.tensor([float(x_i)], requires_grad=True) for x_i in init_vector])

    plt.ion()
    ax = None
    if input_dim==1:
        ax = plt.figure().add_subplot()
        plot_objective_1D(ax, fn, (plot_centre[0]-plot_boundary/2, plot_centre[0]+plot_boundary/2))
    elif input_dim==2:
        ax = plt.figure().add_subplot(projection='3d')
        plot_objective_2D(ax, fn, (plot_centre[0]-plot_boundary/2, plot_centre[0]+plot_boundary/2), (plot_centre[1]-plot_boundary/2, plot_centre[1]+plot_boundary/2))

    optimiser = optimiser(list(x), **kwargs)

    for _ in range(iters):
        objective = fn(*x)

        if input_dim==1:
            plot_point_1D(ax, fn, x[0].item())
        elif input_dim==2:
            plot_point_2D(ax, fn, x[0].item(), x[1].item())

        optimiser.zero_grad()
        objective.backward()
        optimiser.step()
    
    plt.ioff()
    return tuple([x_i.item() for x_i in x])

def optimise_interactive(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda x,y: 0.05 * (x**2 + y**2),
    init_vector: Tuple[float, float] = (-20.0, -19.5),
    plot_centre: Tuple[float, float] = (0.0, 0.0),
    plot_boundary: float = 40.0,
    iters: int = 1000,
    optimisers: Optional[List[Tuple[str, Callable]]] = [("Vanilla SGD", lambda params: torch.optim.SGD(params, lr=0.1)), ("SGD with momentum", lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.99))],
    iter_delay: int = 100,
    gimbal_radius: float = 4.0,
    gimbal_hover: float = 8.0,
    segments: int = 100,
    host: str = "127.0.0.1",
    port: int = 5000,
    open_browser: bool = True,
):
    """
    Launches a local interactive web visualisation of optimisation trajectories
    for a given 2D differentiable function `fn`, comparing multiple optimisers.

    Arguments:
        `fn`: A differentiable function to be minimised. Must take exactly two
        arguments and return a scalar, all of which are `torch.Tensor`s of shape (1,).
        `init_vector`: A tuple of the same dimension as the number of arguments of `fn`, specifying the initial values of the function parameters.
        `plot_centre`: A tuple of the same dimension as the number of arguments of `fn`, specifying the centre point of the plot (in parameter space); defaults to 0 in all dimensions.
        `plot_boundary`: Length of the the plot boundary in all dimensions in the parameter space.
        `iters`: Number of optimiser iterations; defaults to 1000.
        `optimisers`: Optimisation algorithms to use. A list of up to 9 `(name, constructor)` tuples, where each constructor
        takes an iterable of parameters and returns a `torch.optim.Optimizer`
        instance.
        `iter_delay`: Delay (in milliseconds) between each optimiser step.
        `gimbal_radius`: Radius of the circle in which user hovers when gimbal is enabled.
        `gimbal_hover`: Height at which user hovers when gimbal is enabled.
        `segments`: Number of finite elements per axis used to discretise the surface.
        `host`: Host address for the local Flask server.
        `port`: Port for the local Flask server.
        `open_browser`: If True, automatically opens the visualisation in a browser.

    Returns:
        A tuple `(app, server_thread)` where `app` is the Flask application and
        `server_thread` is the background thread running the server.
    """
    assert len(optimisers) <= 9, "Must specify less than 9 optimisers at a time!"

    # Use new parameter names directly (no backward-compat aliases)
    # `plot_centre` is the centre point of the plot (full coordinate tuple)
    # `plot_boundary` is the full width/length of the plotted region
    plot_center = plot_centre
    position_init = init_vector

    # serve static files (index.html) from the package directory so the
    # server works regardless of the current working directory
    pkg_dir = os.path.dirname(__file__)
    app = Flask(__name__, static_folder=pkg_dir, static_url_path='')

    # Surface generation
    def generate_surface(size=plot_boundary, segments_local=segments):
        cx, cz = plot_center
        xs = torch.linspace(cx - size / 2, cx + size / 2, segments_local + 1)
        ys = torch.linspace(cz - size / 2, cz + size / 2, segments_local + 1)

        heights = []
        for y in ys:
            row = []
            for x in xs:
                val = fn(x.unsqueeze(0), y.unsqueeze(0)).item()
                row.append(val)
            heights.append(row)

        return size, segments_local, heights

    # Optimisation runs
    def run_optimisers():
        all_trajs = []

        for name, opt_fn in optimisers:
            x = torch.tensor([position_init[0]], requires_grad=True)
            y = torch.tensor([position_init[1]], requires_grad=True)

            opt = opt_fn([x, y])

            path = []
            grads = []

            for i in range(iters):
                opt.zero_grad()

                loss = fn(x, y)
                loss.backward()

                gx = x.grad.item()
                gy = y.grad.item()
                grads.append([gx, gy])

                path.append({
                    "x": x.item(),
                    "y": loss.item(),
                    "z": y.item(),
                })

                opt.step()

            all_trajs.append({
                "name": name,
                "path": path,
                "grads": grads,
            })

        return all_trajs

    @app.route("/")
    def index():
        return send_from_directory(pkg_dir, 'index.html')

    @app.route('/data')
    def data():
        size, segments_out, heights = generate_surface()
        trajs = run_optimisers()

        return jsonify({
            'size': size,
            'segments': segments_out,
            'heights': heights,
            'plot_centre': list(plot_centre),
            'plot_boundary': plot_boundary,
            'trajectories': trajs,
            'iter_delay': iter_delay,
            'gimbal_radius': gimbal_radius,
            'gimbal_hover': gimbal_hover,
        })

    # run server in a background thread so this function can return
    def run_app():
        # disable reloader when running in thread
        app.run(host=host, port=port, debug=False, use_reloader=False)

    server_thread = threading.Thread(target=run_app)
    server_thread.start()

    # give the server a moment to start
    time.sleep(1.0)

    if open_browser:
        try:
            webbrowser.open(f'http://{host}:{port}/')
        except Exception:
            pass

    return app, server_thread