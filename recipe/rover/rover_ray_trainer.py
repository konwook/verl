# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uuid
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, apply_kl_penalty, compute_response_mask
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import ValidationGenerationsLogger


def compute_rover_centered_rewards(batch: DataProto, reward_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = reward_tensor.sum(dim=-1)
    uids = batch.non_tensor_batch.get("uid")
    if uids is None:
        raise ValueError("ROVER requires non_tensor_batch['uid'] for prompt-level grouping.")
    uids_np = np.asarray(uids)
    rewards_np = rewards.detach().cpu().numpy()

    uniq, inv = np.unique(uids_np, return_inverse=True)
    sums = np.zeros(len(uniq), dtype=np.float32)
    counts = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(sums, inv, rewards_np)
    np.add.at(counts, inv, 1)
    means = sums / np.maximum(counts, 1)

    centered_np = rewards_np - means[inv]
    centered = torch.as_tensor(centered_np, device=reward_tensor.device, dtype=reward_tensor.dtype)
    centered_rewards = centered.unsqueeze(-1) * batch.batch["response_mask"]
    return centered_rewards, centered


class RayROVERTrainer(RayPPOTrainer):
    """
    ROVER trainer that replaces PPO losses with relative-Q regression.
    """

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
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(role_worker_mapping)
        self.use_rm = need_reward_model(role_worker_mapping)
        self.use_critic = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.device_name = device_name if device_name else self.config.trainer.device

        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                # generate sequences
                if not self.async_rollout_mode:
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                else:
                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                gen_batch_output.meta_info.pop("timing", None)

                if self.config.algorithm.adv_estimator == "remax":
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    if not self.async_rollout_mode:
                        gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                    else:
                        gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                    batch = batch.union(gen_baseline_output)
                    rm_scores = None
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        rm_scores = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(rm_scores)
                    reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    keys_to_pop = set(gen_baseline_output.batch.keys())
                    if rm_scores is not None:
                        keys_to_pop.update(rm_scores.batch.keys())
                    batch.pop(batch_keys=list(keys_to_pop))
                    batch.batch["reward_baselines"] = reward_baseline_tensor

                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                batch = batch.union(gen_batch_output)

                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)

                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                reward_extra_infos_dict = {}
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                with torch.no_grad():
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                metrics.update({"actor/entropy": entropy_agg.detach().item()})
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                batch.batch["token_level_scores"] = reward_tensor

                adaptive_lr_scale, adaptive_lr_non_zero_rate = self._compute_adaptive_lr_scale(
                    reward_tensor=reward_tensor,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    batch=batch,
                )
                if adaptive_lr_scale is not None:
                    batch.meta_info["adaptive_lr_scale"] = adaptive_lr_scale
                    batch.meta_info["adaptive_lr_non_zero_rate"] = adaptive_lr_non_zero_rate
                    metrics["actor/adaptive_lr_scale"] = adaptive_lr_scale
                    metrics["actor/adaptive_lr_non_zero_rate"] = adaptive_lr_non_zero_rate

                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                rover_centered_rewards, centered_reward_per_response = compute_rover_centered_rewards(
                    batch, reward_tensor
                )
                batch.batch["rover_centered_rewards"] = rover_centered_rewards
                batch.batch["rover_centered_reward_per_response"] = centered_reward_per_response

                if self.config.trainer.critic_warmup <= self.global_steps:
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    val_metrics = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint()

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
