import copy
from collections import defaultdict

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy import BasePolicy
from tianshou.policy.base import _gae_return
from tianshou.utils import MovAvg, RunningMeanStd
from torch import nn
from torch.nn.functional import one_hot
from torch.nn.utils.clip_grad import clip_grad_norm_


from src.utils.common import get_dist_fn, grad_monitor, last_layer_init, param_monitor, split_batch, weight_init
from src.utils.net import get_actor_critic, get_encoder


# Proximal Policy Optimization (PPO)
class PPOPolicy(BasePolicy):
    def __init__(self, cfg):
        # basic
        super().__init__(cfg.obs_space, cfg.act_space, cfg.p.action_scaling, cfg.p.action_bound_method)

        # model params 
        self.use_rep_for_ac = cfg.use_rep_for_ac  # 是否使用表征用于actor-critic
        self.use_rep_for_im = cfg.use_rep_for_im  # 是否使用表征用于内在动机
        self.device = cfg.device
        self.obs_shape = cfg.obs_space.shape  # 观察空间维度
        self.act_shape = cfg.act_space.shape or cfg.act_space.n  # 动作空间维度

        self.act_dim = np.prod(self.act_shape)
        print(f"obs_shape: {self.obs_shape} | act_shape: {self.act_shape}")
        self.obs_rep_dim = cfg.obs_rep_dim
        self.obs_dim = self.obs_rep_dim if self.use_rep_for_im else self.obs_shape[0]
        self.mlp_hidden_dims = cfg.mlp_hidden_dims
        self.mlp_norm = cfg.mlp_norm


            # 如果指定了prior_dims
        if cfg.p.prior_dims is not None:
            # 情况1: 如果prior_dims的第一个元素小于0
            if cfg.p.prior_dims[0] < 0:
                # 创建从0到obs_dim-1的序列
                self.prior_dims = np.arange(self.obs_dim)
                # 删除指定的负数索引对应的维度
                self.prior_dims = np.delete(self.prior_dims, -np.array(cfg.p.prior_dims))
            # 情况2: 如果prior_dims的元素都是非负数
            else:
                self.prior_dims = cfg.p.prior_dims
            print("prior_dims is ", self.prior_dims)
        else:
            self.prior_dims = None

        self.use_prior_as_input = cfg.p.use_prior_as_input
        self.identity = cfg.p.identity
        self.kwargs_encoder = {
            "obs_shape": self.obs_shape,
            "obs_dim": self.obs_dim if not self.use_prior_as_input else len(self.prior_dims),
            "act_dim": self.act_dim,
            "act_type": self.action_type,
            "mlp_hidden_dims": self.mlp_hidden_dims,
            "obs_rep_dim": self.obs_rep_dim,
            "mlp_norm": self.mlp_norm,
            "device": self.device,
            "use_rep_for_im": self.use_rep_for_im,
            "identity": self.identity,
            "scale": cfg.p.scale,
        }
        self.hidden_dims = cfg.hidden_dims

        # models
        # 当use_rep_for_ac（用于actor-critic的表征）或use_rep_for_im（用于内在动机的表征）为True时，创建编码器
        self.encoder = get_encoder(**self.kwargs_encoder) if self.use_rep_for_ac or self.use_rep_for_im else None
        self._set_ac_input_dim()
        self.actor, self.critic = get_actor_critic(
            self.ac_input_dim, self.hidden_dims, self.act_dim, self.action_type, device=self.device
        )
        self.critics = {"rew_ex": self.critic}
        # rew_ex表示外在奖励(extrinsic reward) 疑问这里为设备么要设置外在奖励
        # 后续可能会添加其他类型的奖励,比如内在奖励(intrinsic reward)
        # 每种奖励都需要对应的critic网络来估计其价值函数

        # init
        # 1. 初始化动作分布的标准差(用于连续动作空间)
        if hasattr(self.actor, "logstd"):  # 检查actor是否有logstd属性
            # 将logstd初始化为一个常数值
            nn.init.constant_(self.actor.logstd, cfg.init_logstd)

        # 2. 对于连续动作空间的actor,特殊初始化最后一层
        if self.action_type == "continuous":
            last_layer_init(self.actor)

        # 3. 初始化critic的最后一层
        last_layer_init(self.critic)
