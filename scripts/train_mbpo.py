#!/usr/bin/envs python3
import numpy as np
import torch
import os
import time
from pytorch_mbpo.pytorch_sac import utils
import hydra

from pytorch_mbpo.pytorch_sac.video import VideoRecorder
from pytorch_mbpo.pytorch_sac.logger import Logger
from pytorch_mbpo.pytorch_sac.replay_buffer import ReplayBuffer

import pytorch_mbpo.envs.termination_fns


def step_fwd_model(fwd_model, obses, actions):
    inputs = np.concatenate((obses, actions), axis=-1)
    pred_samples = fwd_model.sample(inputs)
    rewards = pred_samples[:, :1]
    next_obses = obses + pred_samples[:, 1:]
    return rewards, next_obses


def format_buffer_data(replay_buffer):
    obses, actions, rewards, next_obses, _, _ = replay_buffer.return_all_samples()
    obs_deltas = next_obses - obses
    inputs = np.concatenate((obses, actions), axis=-1)
    targets = np.concatenate((rewards, obs_deltas), axis=-1)
    return inputs, targets


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        cfg.agent.params.obs_dim = obs_dim
        cfg.agent.params.action_dim = action_dim
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        cfg.fwd_model.params.input_dim = obs_dim + action_dim
        if 'feature_dim' in cfg.fwd_model.params.submodule_params.keys():
            cfg.fwd_model.params.submodule_params.feature_dim = 1 + obs_dim
        cfg.fwd_model.params.target_dim = 1 + obs_dim
        self.fwd_model = hydra.utils.instantiate(cfg.fwd_model)
        self.h_sched = [int(i) for i in cfg.h_sched.split(',')]

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

        self.env_buffer = ReplayBuffer(self.env.observation_space.shape,
                                       self.env.action_space.shape,
                                       int(cfg.env_buffer_capacity),
                                       self.device)
        self.model_buffer = ReplayBuffer(self.env.observation_space.shape, self.env.action_space.shape,
                                         self.model_buffer_cap, self.device)

    def evaluate(self):
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            self.video_recorder.save(f'{self.step}.mp4')
            self.logger.log('eval/episode_reward', episode_reward, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # fit forward model, add new model-generated transitions to buffer
            if self.step >= self.cfg.num_seed_steps and self.step % self.cfg.fit_fwd_freq == 0:
                self.fit_fwd_model()
                self.update_model_buffer()
                print(f"[ agent ]  updating agent")

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                sac_metrics = {}
                for i in range(self.cfg.agent_update_freq):
                    trans_batch = list(self.model_buffer.sample(self.agent.batch_size))
                    del trans_batch[-2]  # we don't want the "not_done" field
                    if i % self.agent.critic_update_frequency == 0:
                        sac_metrics = self.agent.update_critic(*trans_batch, None, None)

                    if i % self.agent.actor_update_frequency == 0:
                        sac_metrics.update(self.agent.update_actor_and_alpha(trans_batch[0], None, None))

                    if i % self.agent.critic_target_update_frequency == 0:
                        utils.soft_update_params(self.agent.critic, self.agent.critic_target,
                                                 self.agent.critic_tau)

                if self.step % 2 == 0:
                    self.logger.log_dict(sac_metrics, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.env_buffer.add(obs, action, reward, next_obs, done,
                                done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

    # # # CUSTOM METHODS # # #
    def fit_fwd_model(self):
        train_inputs, train_targets = format_buffer_data(self.env_buffer)
        n, _ = train_inputs.shape
        print(f"[ forward model ]  training on {n} examples")
        fwd_dataset = hydra.utils.instantiate(self.cfg.fwd_dataset)
        fwd_dataset.add_new_data(train_inputs, train_targets)
        fit_metrics = self.fwd_model.fit(fwd_dataset, self.cfg.fwd_model.fit_kwargs)

        # attempt to evaluate model on independent validation dataset
        try:
            root_dir = self.work_dir.split('/exp/')[0]
            data_path = f"{root_dir}/datasets/{self.cfg.task.universe}/{self.cfg.task.env}.dat"  # TODO don't hardcode this
            val_data = torch.load(data_path)
            fit_metrics.update(self.fwd_model.validate(*val_data))
            print(f"[ forward model ]  validation mse - {fit_metrics['val_mse']}")
            print(f"[ forward model ]  validation log_prob - {fit_metrics['val_log_prob']}")
        except FileNotFoundError as e:
            # TODO improve error handling
            print(str(e))
            pass

        for key, value in fit_metrics.items():
            self.logger.log(f"train/fwd_model/{key}", value, self.step)

    def update_model_buffer(self):
        print(f"reallocating model buffer, model buffer capacity: {self.model_buffer_cap}")
        self.model_buffer.reallocate_buffer(self.model_buffer_cap)
        print(f"begin rollouts, rollout_len: {self.rollout_len}")
        for _ in range(self.cfg.num_rollouts):
            batch = self.env_buffer.sample(self.cfg.rollout_batch_size)
            obses = batch[0]
            term_fn = getattr(pytorch_mbpo.envs.termination_fns, self.cfg.task.name)
            for _ in range(self.rollout_len):
                actions = self.agent.actor(obses).sample().cpu().numpy()
                obses = obses.cpu().numpy()
                rewards, next_obses = step_fwd_model(self.fwd_model, obses, actions)
                done_no_maxes = term_fn(obses, actions, next_obses)
                for i in range(obses.shape[0]):
                    transition = (obses[i], actions[i], rewards[i], next_obses[i], 0., done_no_maxes[i])
                    self.model_buffer.add(*transition)

                nonterm_mask = ~done_no_maxes.squeeze(-1)
                if nonterm_mask.sum() <= 1:
                    break
                obses = torch.tensor(next_obses[nonterm_mask], dtype=torch.get_default_dtype())

        print(f"[ forward model ]  rollouts complete, model buffer size - {len(self.model_buffer)}")

    @property
    def rollout_len(self):
        curr_epoch = int(self.step / 1000)
        min_epoch, max_epoch, min_len, max_len = self.h_sched
        assert min_epoch < max_epoch
        assert min_len <= max_len
        epoch_range = max_epoch - min_epoch
        len_range = max_len - min_len
        rollout_len = min_len + len_range * min(max((curr_epoch - min_epoch)/epoch_range, 0), 1)
        return int(rollout_len)

    @property
    def model_buffer_cap(self):
        return self.cfg.rollout_batch_size * self.rollout_len * self.cfg.rollouts_retained


@hydra.main(config_path=f"../config/train_mbpo.yaml", strict=True)
def main(cfg):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(cfg.pretty())
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
