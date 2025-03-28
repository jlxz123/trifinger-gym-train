# nohup python trifinger_gym_train/leibnizgym/scripts/m_rlg_hydra.py > /data/user/wanqiang/zbf_downloads/trained_logs/output_2025.3.22.log 2>&1 &
# nohup python trifinger_gym_train/leibnizgym/scripts/m_rlg_hydra.py --play_or_not --checkpoint "/data/user/wanqiang/zbf_downloads/trained_models/PPO_continuous_itera1560.pth"> /data/user/wanqiang/zbf_downloads/trained_logs/test_2025.3.25.log 2>&1 &

import hydra
import os
import logging
import tyro
from omegaconf import DictConfig, OmegaConf, open_dict
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Dict, List, Any
from hydra.core.config_store import ConfigStore
from m_rlg_train import run_rlg_hydra
from pathlib import Path

PATH_ROOT = Path(__file__).parents[1].absolute()
PATH_CFG = PATH_ROOT / "../resources/config"



@dataclass
class SimConfig:
    """Configuration for the IsaacGym simulator."""

    dt: float = 0.05
    substeps: int = 4
    up_axis: str = "z"
    use_gpu_pipeline: bool = MISSING
    num_client_threads: int = 0
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    physx: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_threads": 4,
            "solver_type": 1,
            "use_gpu": False,  # set to False to run on CPU
            "num_position_iterations": 3,
            "num_velocity_iterations": 0,
            "contact_offset": 0.002,
            "rest_offset": 0.0,
            "bounce_threshold_velocity": 0.5,
            "max_depenetration_velocity": 1000.0,
            "default_buffer_size_multiplier": 5.0,
        }
    )
    flex: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_outer_iterations": 5,
            "num_inner_iterations": 20,
            "warm_start": 0.8,
            "relaxation": 0.75,
        }
    )


@dataclass
class EnvConfig:
    """Configuration for all instances of `EnvBase`."""

    env_name: str = MISSING
    # general env settings
    num_instances: int = MISSING
    seed: int = MISSING
    spacing: float = 1.0
    aggregate_mode: bool = True
    # command settings
    control_decimation: int = 1
    # physics settings
    physics_engine: str = MISSING
    sim: SimConfig = SimConfig()


@dataclass
class Trifinger(EnvConfig):
    """Configuration for all instances of `TrifingerEnv`."""

    env_name: str = "Trifinger"
    episode_length: int = 750
    task_difficulty: int = MISSING
    enable_ft_sensors: bool = False
    # observation settings
    asymmetric_obs: bool = False
    normalize_obs: bool = True
    # reset distribution settings
    apply_safety_damping: bool = False  # True
    command_mode: str = "position"
    normalize_action: bool = True
    reset_distribution: Dict[str, Any] = field(
        default_factory=lambda: {
            "object_initial_state": {"type": "random"},
            "robot_initial_state": {
                "dof_pos_stddev": 0.4,
                "dof_vel_stddev": 0.2,
                "type": "default",
            },
        }
    )
    # reward settings
    reward_terms: Dict[str, Any] = field(
        default_factory=lambda: {
            "finger_move_penalty": {
                "activate": False,  # True
                "weight": -0.05,  # -0.1,
            },
            "finger_reach_object_rate": {
                "activate": True,
                "norm_p": 2,
                "weight": -27.5,  # -175
                # "thresh_sched_start":0,
                # "thresh_sched_end":5e3,
            },
            "object_dist": {
                "activate": True,
                "weight": 2000,  # 2000,   //2000 max 2.5
            },
            "object_rot": {
                "activate": False,  # False
                "weight": 300,
            },
            "object_rot_delta": {"activate": False, "weight": -250},
            "object_move": {
                "activate": True,  # False,
                "weight": -75,  # -750,
            },
            "finger_reach_object_dist": {
                "activate": False,
                "weight": 5000,
                "norm_p": 2,
            },
        }
    )
    # # termination settings
    termination_conditions: Dict[str, Any] = field(
        default_factory=lambda: {
            "success": {
                "activate": True,  # False,
                "bonus": 5000,  # 5000.0,
                "orientation_tolerance": 0.1,  # rad
                "position_tolerance": 0.02,  # m
            }
        }
    )


@dataclass
class TrifingerDifficulty1(Trifinger):
    """Trifinger Difficulty 1 Configuration."""

    task_difficulty = 1


@dataclass
class TrifingerDifficulty2(Trifinger):
    """Trifinger Difficulty 2 Configuration."""

    task_difficulty = 2


@dataclass
class TrifingerDifficulty3(Trifinger):
    """Trifinger Difficulty 3 Configuration."""

    task_difficulty = 3


@dataclass
class TrifingerDifficulty4(Trifinger):
    """Mode for testing to try to get the rotation reward up and running."""

    task_difficulty = 4
    episode_length = 750
    reward_terms: Dict[str, Any] = field(
        default_factory=lambda: {
            "finger_move_penalty": {
                "activate": False,  # True,
                "weight": -0.1,
            },
            "finger_reach_object_rate": {
                "activate": True,
                "norm_p": 2,
                "weight": -250,
                "thresh_sched_start": 0,
                "thresh_sched_end": 1e7,
            },
            "object_dist": {
                "activate": True,
                "weight": 2000,
                "thresh_sched_start": 0,
                "thresh_sched_end": 10e10,
            },
            "object_rot": {
                "activate": False,
                "weight": 2000,
                "epsilon": 0.01,
                "scale": 3.0,
                "thresh_sched_start": 1e7,
                "thresh_sched_end": 1e10,  # dont end!
            },
            "object_rot_delta": {"activate": False, "weight": -250},
            "object_move": {
                "activate": False,
                "weight": -750,
            },
        }
    )
    termination_conditions: Dict[str, Any] = field(
        default_factory=lambda: {
            "success": {
                "activate": True,
                "bonus": 5000.0,
                "orientation_tolerance": 0.25,
                "position_tolerance": 0.02,
            }
        }
    )


@dataclass
class RLGConfig:
    """Configuration for RLGames."""

    asymmetric_obs: bool = (
        MISSING  # argument specifying whether this config requires asymmetric states
    )
    params: Dict[Any, Any] = MISSING


@dataclass
class RLGArgs:
    verbose: bool = False


@dataclass
class Args:
    """Items which must be propogated down to other areas of configuration or which are expected as args in other places in the config."""

    # configuration about the env creation
    cfg_env: str = "Base"
    cfg_train: str = "Base"
    # todo - maybe do some validation on this? should match up to the config selected
    task: str = "Trifinger"
    task_type: str = "Python"  # Choose Python or C++
    experiment_name: str = "Base"  # used in RLGames

    # confuguration about the env itself
    num_envs: int = 256  # overrides the default number of envs
    # todo - investigate interaction between this and pyDR
    randomize: bool = False  # whether to apply physics domain randomisation

    # other misc rags
    seed: int = 7  # random seed
    verbose: bool = False
    logdir: str = "logs/"  # backs up the configs for this run

    # devcie config
    physics_engine: Any = (
        "physx"  # field(default_factory=lambda: gymapi.SIM_PHYSX) # 'physx' or 'flex'
    )
    device: str = "GPU"  # CPU or GPU for running physics
    ppo_device: str = "GPU"  # whether to use GPU for inference with PPO

    # RLGames Arguments
    play: bool = False  # if set runs trained policy (for use with rl games)
    train: bool = MISSING  # opposite of play
    checkpoint: str = ""  # used to set checkpoint path

    # Common Gym Arguments
    headless: bool = False  # disables rendering
    compute_device_id: int = 0  # for CUDA
    graphics_deice_id: int = 0  # graphics device id

    wandb_project_name: str = "trifinger-manip"
    wandb_log: bool = True
    offline: bool = False


@dataclass
class Config:
    """Base config class."""

    gym: EnvConfig = MISSING
    rlg: Dict[str, Any] = MISSING
    args: Args = Args()

    output_root: str = MISSING


@dataclass
class ConfigSlurm:
    """Base config class."""

    gym: EnvConfig = MISSING
    rlg: Dict[str, Any] = MISSING
    args: Args = Args()


def update_cfg(cfg):
    """Modifies cfg by copying key arguments to the correct places.

    Args:
        cfg: Hydra config to modify
    """

    cfg.args.train = not cfg.args.play
    # Override settings if passed by the command line
    # Override number of environments
    cfg.gym.num_instances = cfg.args.num_envs
    # Override the phyiscs settings
    cfg.gym.sim.use_gpu_pipeline = cfg.args.device == "GPU"
    cfg.gym.sim.physx.use_gpu = cfg.args.device == "GPU"
    cfg.gym.physics_engine = cfg.args.physics_engine

    # Set cfg to enable asymmetric training
    cfg.gym.asymmetric_obs = cfg.rlg.asymmetric_obs

    # %% RLG config
    #  Override experiment name
    if cfg.args.experiment_name != "Base":
        cfg.rlg.params.config.name = f"{cfg.args.experiment_name}_{cfg.args.task_type}_{cfg.args.device}_{str(cfg.args.physics_engine).split('_')[-1]}"

    cfg.rlg.params.load_checkpoint = cfg.args.checkpoint != ""
    cfg.rlg.params.load_path = cfg.args.checkpoint

    # Set number of environment instances
    with open_dict(cfg):
        cfg.rlg["params"]["config"]["minibatch_size"] = cfg.args.num_envs
        cfg.rlg["params"]["config"]["num_actors"] = cfg.args.num_envs
        # Set minibatch size for central value config
        if "central_value_config" in cfg.rlg["params"]["config"]:
            cfg.rlg["params"]["config"]["central_value_config"]["minibatch_size"] = (
                cfg.args.num_envs
            )
        cfg.gym.seed = cfg.args.seed
        cfg.rlg.seed = cfg.args.seed


def get_config_store():
    # Instantiates the different configurations in the correct groups.
    cs = ConfigStore.instance()
    cs.store(group="gym", name="trifinger_difficulty_1", node=TrifingerDifficulty1)
    cs.store(group="gym", name="trifinger_difficulty_2", node=TrifingerDifficulty2)
    cs.store(group="gym", name="trifinger_difficulty_3", node=TrifingerDifficulty3)
    cs.store(group="gym", name="trifinger_difficulty_4", node=TrifingerDifficulty4)

    # Don't need to instantiate the RLG configs as they are still yaml's - see corresponding directory.
    cs.store(name="config", node=Config)


@dataclass
class CustomArg:
    train_algo: str = "ppo"
    test_algo: str = "ppo"
    play_or_not: bool = False
    checkpoint: str = ""
    num_envs: int = 1024


def launch_rlg_hydra(arg: CustomArg):
    log = logging.getLogger(__name__)

    # 手动初始化 Hydra 并加载默认配置
    with hydra.initialize(config_path="../../resources/config"):  # str(PATH_CFG)
        cfg = hydra.compose(config_name="config")

    # 更新配置
    cfg.args.play = arg.play_or_not
    cfg.train = arg.train_algo
    cfg.test = arg.test_algo
    if cfg.args.play:
        if cfg.train not in ["ppo", "sac", "sacn", "ddpg", "td3", "cql"]:
            raise ValueError(f"无效的算法名称: {cfg.train}. 请在 ['ppo', 'sac', 'sacn', 'ddpg', 'td3', 'cql'] 中选择。")
        cfg.args.num_envs = 1
        if cfg.train == "cql":
            cfg.args.offline = True
    else:
        if cfg.test not in ["ppo", "sac", "sacn", "ddpg", "td3", "cql"]:
            raise ValueError(f"无效的算法名称: {cfg.train}. 请在 ['ppo', 'sac', 'sacn', 'ddpg', 'td3', 'cql'] 中选择。")

        cfg.args.num_envs = arg.num_envs
        if cfg.test == "cql":
            cfg.args.offline = True
    cfg.gym.task_difficulty = 1
    cfg.args.headless = True
    cfg.args.wandb_log = False
    cfg.gym.control_decimation = 30
    cfg.args.checkpoint = arg.checkpoint


    update_cfg(cfg)

    print(cfg.train)
    print(cfg.test)
    run_rlg_hydra(cfg)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    cs = get_config_store()
    # os.environ["WANDB_API_KEY"] = 'c01244c0564fc20f4feefbc17ef4a666ff8a73cd'
    # os.environ["WANDB_MODE"] = "offline"
    customarg = tyro.cli(CustomArg)
    launch_rlg_hydra(customarg)
