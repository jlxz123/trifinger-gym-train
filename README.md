# 三指机器人仿真训练测试环境



## 项目简介
在项目 [Leibniz Gym](https://github.com/pairlab/leibnizgym) 基础上，参考项目 [leibnizgym_copy](https://github.com/wqqqqqqw/leibnizgym_copy) 和项目 [tri-fingers](https://github.com/wty-yy/tri-fingers) ，并在项目 [trifinger-gym-train](https://github.com/wty-yy/trifinger-gym-train) 基础上做了一些修改，构建较为完善的三指机器人仿真训练及测试环境，支持 `ppo` 、 `sac` 、 `cql` 、 `ddpg` 、 `td3` 等多个强化学习算法。



## 修改内容


### `rlg_hydra.py`
增加'CustomArg'类，定义了一些必要的参数调整选项，同时结合tyro库，可以在命令行用指令进行具体参数值设置，内容如下：
```python
@dataclass
class CustomArg:
    # 训练和测试算法，可从ppo、sac、sacn、ddpg、cql等算法中选择
    train_algo: str = "ppo"
    test_algo: str = "ppo"
    # 进行训练/测试，默认为 False （训练模式），设置 True 则进行测试
    play_or_not: bool = False
    # 以训练好的模型路径，用于继续训练或者进行测试
    checkpoint: str = ""
    # 训练时运行的实例数量，play_or_not 设置为 True 时该项无效
    num_envs: int = 1024
```
具体用法见[tyro官方网站](https://pypi.org/project/tyro/)。


### `rlg_train.py`
删除了不必要的函数 `run_rlg()` ，并对前面的库函数调用部分做了修改，将函数 `parse_vec_task` 和函数 `create_rlgpu_env2` 移到新文件 `rlg_env.py` （同级文件夹）中，避免了文件重复调用问题。


### `ppo.py` 、 `sac.py` 、 `ddpg.py` 、 `td3.py`
完善测试、训练流程，checkpoint保存位置，优化代码写法。

