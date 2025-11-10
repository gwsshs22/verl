import logging
import os

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import (
    log_gpu_memory_usage,
)
from verl.utils.device import get_device_name, get_torch_device
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.megatron_workers import ActorRolloutRefWorker as ARRWorker
from verl.workers.megatron_workers import CriticWorker, RewardModelWorker
from verl.workers.rollout import get_rollout_class


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["ActorRolloutRefWorker", "CriticWorker", "RewardModelWorker", "RolloutWorker"]

class ActorRolloutRefWorker(ARRWorker):
    def __init__(self, config: DictConfig, role: str):
        assert role in ["actor", "ref"]
        tmp_role = "ref" if role == "ref" else "actor_rollout"
        super().__init__(config, tmp_role)
        if role == "actor":
            self._is_rollout = False
        self.role = role

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info

        params_generator = self._get_actor_params_generator()
        ret = []
        for key, tensor in params_generator:
            ret.append((key, tensor.size(), tensor.dtype))

        self._weights_info = ret
        return ret

    def _get_actor_params_generator(self):
        assert self._is_actor
        from verl.models.mcore import get_mcore_weight_converter
        from verl.utils.megatron_utils import per_tensor_generator

        layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
        generator = per_tensor_generator(
            self.actor.actor_module,
            self.actor_model_config,
            weight_converter,
            self.tf_config,
            layer_name_mapping,
        )
        return generator

class RolloutWorker(ActorRolloutRefWorker):

    def __init__(self, config: DictConfig, role: str):
        assert role == "rollout"
        ARRWorker.__init__(self, config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_transformer_config = {}
        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        trust_remote_code = self.config.model.get("trust_remote_code", False)

        from verl.utils.model import get_generation_config

        self._init_hf_config_and_tf_config(
            self.config.model.path,
            self.config.model.path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            trust_remote_code,
        )
        self.generation_config = get_generation_config(self.local_path)

        from torch.distributed.device_mesh import init_device_mesh

        assert self.config.rollout.name == "vllm"
        assert self.config.rollout.mode == "sync"


        # NOTE(sgm): If the QKV and gate_up projection layer are concate together in actor,
        # we will reorganize their weight format when resharding from actor to rollout.

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            get_device_name(), mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
        self._register_dispatch_collect_info(
            "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )
        log_gpu_memory_usage("Before building vllm rollout", logger=None)

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage("After building vllm rollout", logger=logger)

        from .vllm_sharding_manager import VLLMShardingManager
        sharding_manager = VLLMShardingManager(
            inference_engine=rollout.inference_engine,
            device_mesh=rollout_device_mesh,
        )

        log_gpu_memory_usage("After building sharding manager", logger=logger)

        self.rollout, self.sharding_manager = rollout, sharding_manager
        self.rollout.sharding_manager = sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info