import uuid
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils.tracking import ValidationGenerationsLogger
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.debug import marked_timer

from recipe.streamrl.utils import need_critic

class StreamrlRayTrainer(RayPPOTrainer):

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        collate_fn=None,
        train_sampler: Sampler | None = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        for role, role_name in [(Role.Actor, "actor"), (Role.Rollout, "rollout")]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=role_name,
            )
            self.resource_pool_to_cls[resource_pool][role_name] = role_cls

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls
        

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "steps")
            assert (
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                is not None
            ), "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.actor_wg = all_wg["actor"]
        self.rollout_wg = all_wg["rollout"]
        self.actor_wg.init_model()
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified
        weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(weights_info)

        self.create_weight_sync_group()
        self.sync_rollout_weights()

    def create_weight_sync_group(self):
        master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote())
        master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
        world_size = len(self.actor_wg.workers + self.rollout_wg.workers)
        self.actor_wg.create_weight_sync_group(
            master_address,
            master_port,
            0,
            world_size,
        )
        ray.get(
            self.rollout_wg.create_weight_sync_group(
                master_address,
                master_port,
                len(self.actor_wg.workers),
                world_size,
            )
        )

    def sync_rollout_weights(self):
        if not self.hybrid_engine:
            self.actor_wg.sync_rollout_weights()
            ray.get(self.rollout_wg.sync_rollout_weights())

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict


    def _async_gen_next_batch(self, continuous_iterator):
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
        try:
            epoch, batch_dict = next(continuous_iterator)
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error in async_gen_next_batch: {e}")
            return None

        # Create the initial batch from the data loader
        batch = DataProto.from_single_dict(batch_dict)

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        # sync weights from actor to rollout
        self.sync_rollout_weights()

        # async generation
        gen_batch_output = self.rollout_wg.async_generate_sequences(gen_batch)

        # Launch individual reward computations as each generation completes
        future_reward = None
        if self.config.reward_model.launch_reward_fn_async:
            # Store the object reference and set up callback
            future_reward = self._launch_individual_rewards.remote(
                gen_batch_output, self.config, self.tokenizer, batch.non_tensor_batch
            )

        # Return the original, now-modified `batch` and the `future_reward`
        return GenerationBatchFuture(epoch, batch, gen_batch_output, future_reward)

    @staticmethod
    @ray.remote
    def _launch_individual_rewards(gen_batch_output, config, tokenizer, original_non_tensor_batch):
        # Get generation results
        gen_batch_result = gen_batch_output.get()

        # Repeat non_tensor_batch to match the number of responses
        n = config.actor_rollout_ref.rollout.n
        repeated_non_tensor_batch = {}
        for key, value in original_non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(value, n, axis=0)

        # Split into individual responses with preserved non_tensor_batch
        responses_split = []
        for i in range(len(gen_batch_result)):
            response_data = gen_batch_result[i : i + 1]  # Get single response
            # Add repeated non_tensor_batch values
            for key in repeated_non_tensor_batch:
                response_data.non_tensor_batch[key] = repeated_non_tensor_batch[key][i : i + 1]
            responses_split.append(response_data)

        # Launch async reward computation
        reward_futures = [
            compute_reward_async.remote(response_data, config, tokenizer) for response_data in responses_split
        ]

        # Wait for results and combine
        results = ray.get(reward_futures)
        rewards_list = [r[0] for r in results]
        extras_list = [r[1] for r in results]

        combined_reward_tensor = torch.cat(rewards_list, dim=0)
        combined_extras_dict = {}
        if extras_list and extras_list[0]:
            for key in extras_list[0].keys():
                combined_extras_dict[key] = [d[key] for d in extras_list if key in d]

        return combined_reward_tensor, combined_extras_dict

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        
        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        # across epoch iterator
        continuous_iterator = self._create_continuous_iterator()
        batch_data_future = None
        is_first_step = True
        while batch_data_future is not None or is_first_step:
            is_first_step = False
            do_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )
            if do_profile:
                self.actor_wg.start_profile()
                if not self.hybrid_engine:
                    self.rollout_wg.start_profile()
                if self.use_reference_policy:
                    self.ref_policy_wg.start_profile()
                if self.use_critic:
                    self.critic_wg.start_profile()
                if self.use_rm:
                    self.rm_wg.start_profile()

            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
                    batch_data_future = self._async_gen_next_batch(continuous_iterator)

                # wait for the previous batch
                with marked_timer("wait_gen", timing_raw, color="red"):
                    epoch, batch, gen_batch_output, future_reward = batch_data_future.get()
                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)
                
                return

class GenerationBatchFuture:
    """
    Wrapper class for encapsulating batch generation results
    """

    def __init__(self, epoch, batch, gen_batch_output, future_reward=None):
        """
        :param epoch: current epoch
        :param batch: Input batch data
        :param gen_batch_output: Generated sequences from the main model (DataProtoFuture)
        :param future_reward: Future for reward computation (optional)
        """
        self.epoch = epoch
        self.batch = batch
        self.gen_batch_output = gen_batch_output
        self.future_reward = future_reward

    def get(self):
        """
        Get the actual results by calling get() method on gen_batch_output

        Returns:
            tuple: (epoch, batch, gen_batch_result, future_reward)
                - epoch: Current epoch
                - batch: Original input batch data
                - gen_batch_result: Result from gen_batch_output.get() or gen_batch_output itself
                - future_reward: Future for reward computation if available, else None
        """
        # Call get() method on gen_batch_output if available
        if hasattr(self.gen_batch_output, "get"):
            gen_batch_result = self.gen_batch_output.get()
        else:
            gen_batch_result = self.gen_batch_output

        return self.epoch, self.batch, gen_batch_result, self.future_reward