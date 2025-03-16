## 使用方法
[assets/history_command_by_wanqiang.md](./assets/history_command_by_wanqiang.md)中记录了之前wq使用的命令，在安装`trifinger-gym`后

### 不优美的算法启动方法
在`leibnizgym/utils/rlg_train.py`中修改

完成上启动执行如下：
```bash
conda activate rlgpu
# 指定第1张显卡
CUDA_VISIBLE_DEVICES=0 python leibnizgym/scripts/rlg_hydra.py
```
训练的日志会保存在[output/](./output/)文件夹对应日期下

修改训练算法，在`leibnizgym/scripts/rlg_hydra.py`中的
```python
# 314行内容
# 指定train和test算法
# 支持ppo_tt, cql, sacn, sac, ddpg, td3，将下面算法修改即可
OmegaConf.update(cfg, "train", "sac")
OmegaConf.update(cfg, "test", "sac")
```

## 更新日志
### 2025.3.16.
重新整理代码仓库，并在此记录后续更新日志，之前训练过的model和output可以在`/data/user/wanqiang/document/leibnizgym-wq`中找到