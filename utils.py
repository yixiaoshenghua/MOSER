import os
import pickle
import torch
from torch import nn
import numpy as np 
import moviepy.editor as mpy
import re
import matplotlib.pyplot as plt 
from typing import Iterable
from torch.nn import Module
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def preprocess_obs(obs):
    obs = obs.to(torch.float32)/255.0 - 0.5
    return obs

def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
          output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):

        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]

class Logger:

    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, step):
        for key, value in scalar_dict.items():
            print('{} : {}'.format(key, value))
            self.log_scalar(value, key, step)
        self.dump_scalars_to_pickle(scalar_dict, step)

    def log_videos(self, videos, step, max_videos_to_save=1, fps=20, video_title='video', writer=False, store=False):
        if store:
            # max rollout length
            max_videos_to_save = np.min([max_videos_to_save, videos.shape[0]])
            max_length = videos[0].shape[0]
            for i in range(max_videos_to_save):
                if videos[i].shape[0]>max_length:
                    max_length = videos[i].shape[0]

            # pad rollouts to all be same length
            for i in range(max_videos_to_save):
                if videos[i].shape[0]<max_length:
                    padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
                    videos[i] = np.concatenate([videos[i], padding], 0)

                clip = mpy.ImageSequenceClip(list(videos[i]), fps=fps)
                new_video_title = video_title+'{}_{}'.format(step, i) + '.gif'
                filename = os.path.join(self._log_dir, new_video_title)
                clip.write_gif(filename, fps=fps)
        
        if writer:
            self._summ_writer.add_video(video_title, videos, step, fps)

    def log_args(self, args):
        self._summ_writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    def dump_scalars_to_pickle(self, metrics, step, log_title=None):
        log_path = os.path.join(self._log_dir, "scalar_data.pkl" if log_title is None else log_title)
        with open(log_path, 'ab') as f:
            pickle.dump({'step': step, **dict(metrics)}, f)

    def flush(self):
        self._summ_writer.flush()

class AvgDict:
    def __init__(self):
        self.log_dict = OrderedDict()
        self.curr_step = 0

    def update(self, items):
        self.log_dict.update(items)

    def record(self, items):
        for k, v in items.items():
            if self.curr_step == 0:
                self.log_dict[k] = [v]
            else:
                self.log_dict[k].append(v)
        self.curr_step += 1

    def dump(self):
        for k in self.log_dict.keys():
            if type(self.log_dict[k]) == list:
                self.log_dict[k] = np.mean(self.log_dict[k])
        self.curr_step = 0

    @property     
    def dict(self):
        self.dump()
        return self.log_dict

def compute_return(rewards, values, discounts, td_lam, last_value):

    next_values = torch.cat([values[1:], last_value.unsqueeze(0)],0)  
    targets = rewards + discounts * next_values * (1-td_lam)
    rets =[]
    last_rew = last_value

    for t in range(rewards.shape[0]-1, -1, -1):
        last_rew = targets[t] + discounts[t] * td_lam *(last_rew)
        rets.append(last_rew)

    returns = torch.flip(torch.stack(rets), [0])
    return returns

def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        # step = tf.cast(step, tf.float32) #Fixme cast
        match = re.match(r'linear\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r'warmup\((.+),(.+)\)', string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.clamp(step / warmup, 0, 1)
            return scale * value
        match = re.match(r'exp\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r'horizon\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)