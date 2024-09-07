import torch
import numpy as np


class ReplayBuffer:

    def __init__(self, size, obs_shape, action_size, seq_len, batch_size, sensor_action_size=3):

        self.size = size
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.idx = 0
        self.sort_reward_idxs = None
        self.full = False
        self.observations = np.empty((size, *obs_shape), dtype=np.uint8) 
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.sensor_actions = np.empty((size, sensor_action_size), dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32) 
        self.terminals = np.empty((size,), dtype=np.float32)
        self.steps, self.episodes = 0, 0
    
    def add(self, obs, ac, rew, done, sensor_ac=None):

        self.observations[self.idx] = obs['image']
        self.actions[self.idx] = ac
        self.no_sensor_ac = False
        if sensor_ac is not None:
            self.sensor_actions[self.idx] = sensor_ac
        else:
            self.no_sensor_ac = True
        self.rewards[self.idx] = rew
        self.terminals[self.idx] = done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1 
        self.episodes = self.episodes + (1 if done else 0)

    def _sample_idx(self, L):

        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = not self.idx in idxs[1:] 
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        observations = self.observations[vec_idxs]
        if self.no_sensor_ac:
            return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.terminals[vec_idxs].reshape(L, n)
        else:
            return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.terminals[vec_idxs].reshape(L, n), self.sensor_actions[vec_idxs].reshape(L, n, -1)

    def sample(self):
        n = self.batch_size
        l = self.seq_len
        if self.no_sensor_ac:
            obs, acs, rews, terms = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
            return obs, acs, rews, terms
        else:
            obs, acs, rews, terms, sensor_acs = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
            return obs, acs, rews, terms, sensor_acs

    def add_from_files(self, dirpath):
        import os
        paths = os.listdir(dirpath)
        for path in paths:
            data = np.load(os.path.join(dirpath, path)) # 'image': (bs, img_size, img_size, 3), 'action': (bs, ac), 'reward': (bs,), 'discount': (bs,)
            size = data['image'].shape[0]
            self.observations[self.idx:self.idx+size] = data['image'].transpose(0, 3, 1, 2) if data['image'].shape[-1] == 3 else data['image']
            self.actions[self.idx:self.idx+size] = data['action']
            self.rewards[self.idx:self.idx+size] = data['reward']
            if 'done' in data.files:
                self.terminals[self.idx:self.idx+size] = data['done']
            else:
                self.terminals[self.idx:self.idx+size] = np.concatenate([np.zeros((size-1)), np.ones((1))])
            self.idx = (self.idx + size) % self.size
            self.full = self.full or self.idx == 0
            # self.steps += size
            self.episodes += 1


class SensorBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size=1024):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones