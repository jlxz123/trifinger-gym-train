# leibnizgym
from leibnizgym.wrappers.vec_task import VecTaskPython
from leibnizgym.utils.errors import InvalidTaskNameError
from leibnizgym.utils.message import *

# rl-games
from rl_games.common import wrappers

# python
import os
import argparse


def parse_vec_task(args: argparse.Namespace, cfg: dict) -> VecTaskPython:
    """Parses the configuration parameters for the environment task.

    TODO (@mayank): Remove requirement for args and make this a normal function
                    inside utils.
    Args:
        args: command line arguments.
        cfg: environment configuration dictionary (task)

    Returns:
        TThe vectorized RL-env wrapped around the task.
    """
    # create native task and pass custom config
    if args.task_type == "Python":
        # check device on which to run agent and environment
        if args.device == "CPU":
            print_info("Running using python CPU...")
            # check if agent is on different device
            sim_device = "cpu"
            ppo_device = "cuda:0" if args.ppo_device == "GPU" else "cpu"
        else:
            print_info("Running using python GPU...")
            sim_device = "cuda:0"
            ppo_device = "cuda:0"
        # create the IsaacEnvBase defined using leibnizgym
        try:
            task = eval(args.task)(
                config=cfg,
                device=sim_device,
                visualize=not args.headless,
                verbose=args.verbose,
            )
        except NameError:
            raise InvalidTaskNameError(args.task)
        # wrap environment around vec-python wrapper
        env = VecTaskPython(task, rl_device=ppo_device, clip_obs=5, clip_actions=1)
    else:
        raise ValueError(f"No task of type `{args.task_type}` in leibnizgym.")

    return env


def create_rlgpu_env(**kwargs):
    """
    Creates the task from configurations and wraps it using RL-games wrappers if required.
    """
    # print(kwargs)
    # TODO (@arthur): leibnizgym parse task
    env = parse_vec_task(kwargs["cli_args"], kwargs["task_cfg"])
    # print the environment information
    print_info(env)
    # save environment config into file
    logdir = kwargs["cli_args"]["logdir"]
    env.dump_config(os.path.join(logdir, "env_config.yaml"))
    # wrap around the environment
    frames = kwargs.pop("frames", 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env
