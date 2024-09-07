import gym
import os
import sys
import torch
import random
from dm_control.mujoco.engine import Camera, MovableCamera
import numpy as np
from PIL import Image
import skvideo.io
import cv2
import tqdm

DMC_ENV = ['cheetah', 'walker', 'quadruped', 'reacher']
JACO_ENV = ['reach']
GYM_ROBOT_ENV = ['Fetch']

def make_env(args):
    if args.env.split('-')[0] in DMC_ENV+JACO_ENV:
        if args.env.split('-')[-1] == 'fixed':
            env = FixedCameraDeepMindControl(args, '-'.join(args.env.split('-')[:2]), args.seed)
        elif args.env.split('-')[-1] == 'changed':
            env = ChangeCameraDeepMindControl(args, '-'.join(args.env.split('-')[:2]), args.seed)
        elif args.env.split('-')[-1] == 'multiview':
            domain, task = args.env.split('-')[:2]
            if domain in DMC_ENV:
                env = MultiViewDeepMindControl(args, '-'.join(args.env.split('-')[:2]), args.seed)
            elif domain in JACO_ENV:
                env = MultiViewJaco(args, '-'.join(args.env.split('-')[:2]), args.seed)
        else:
            env = DeepMindControl(args, '-'.join(args.env.split('-')[:2]), args.seed, camera=args.camera_id)
    elif args.env.split('-')[0] in GYM_ROBOT_ENV:
        env = GymRobotEnv(args, '-'.join(args.env.split('-')[:2]), args.seed, camera=args.camera_id)
    env = ActionRepeat(env, args.action_repeat)
    env = NormalizeActions(env)
    env = TimeLimit(env, args.time_limit / args.action_repeat)
    #env = RewardObs(env)
    return env

class GymChangeCameraWrapper(gym.Wrapper):
    '''
    Discrete camera view version, 
    which means the controller action is also needed to be discretized
    '''
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        _ = env.reset()
        _ = env.render(mode='rgb_array')
        self.viewer_cam_distance = self.get_viewer_distance()
        self.viewer_cam_azimuth = self.get_viewer_azimuth()
        self.viewer_cam_elevation = self.get_viewer_elevation()
        print("distance: {}, azimuth: {}, elevation: {}".format(self.viewer_cam_distance, self.viewer_cam_azimuth, self.viewer_cam_elevation))

    def get_viewer_distance(self):
        return self.viewer.cam.distance
    
    def get_viewer_azimuth(self):
        return self.viewer.cam.azimuth
    
    def get_viewer_elevation(self):
        return self.viewer.cam.elevation

    def viewer_setup(self, body_name='link7', delta_distance=0.0, delta_azimuth=0., delta_elevation=0.):

        body_id = self.sim.model.body_name2id(body_name)
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance += delta_distance
        self.viewer.cam.azimuth += delta_azimuth
        self.viewer.cam.elevation += delta_elevation

    def reset(self):
        obs = self.env.reset()
        self.viewer.cam.distance = self.viewer_cam_distance
        self.viewer.cam.azimuth = self.viewer_cam_azimuth
        self.viewer.cam.elevation = self.viewer_cam_elevation
        return self._get_obs()

    def step(self, action):
        '''
        action consists of original actions and view-change action
        '''
        motor_action, sensor_action = action[:-3], action[-3:]
        delta_distance, delta_azimuth, delta_elevation = sensor_action
        self.viewer.cam.distance += delta_distance
        self.viewer.cam.azimuth += delta_azimuth
        self.viewer.cam.elevation += delta_elevation
        _, reward, done, info = self.env.step(motor_action)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = self.env.render(mode='rgb_array')#.transpose(2, 0, 1)
        return obs


class GymRobotEnv:
    def __init__(self, args, name, seed, size=(64, 64), camera=None):
        self.args = args
        domain, task = name.split('-', 1)
        self._env = gym.make(domain+task+'-v1')
        self._env.seed(seed)
        self._size = size
        self._camera = camera
        self._steps = 0

    @property
    def observation_space(self):
        spaces = {}
        spaces['image'] = gym.spaces.Box(
            0, 255, (3,) + self._size , dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        self._steps += 1
        _, reward, done, info = self._env.step(action)
        obs = dict()
        obs['image'] = self._env.render('rgb_array', width=self._size[1], height=self._size[0]).transpose(2, 0, 1).copy()
        return obs, reward, done, info
    
    def reset(self):
        self._steps = 0
        state = self._env.reset()
        obs = dict()
        obs['image'] = self._env.render('rgb_array', width=self._size[1], height=self._size[0]).transpose(2, 0, 1).copy()
        return obs

class MultiViewDeepMindControl:

    def __init__(self, args, name, seed, size=(64, 64), cameras=[0, 1]):
        self.args = args
        domain, task = name.split('-', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            if domain in DMC_ENV:
                from dm_control import suite
                self._env = suite.load(domain, task, task_kwargs={'random':seed})
            elif domain in JACO_ENV:
                self._env = make_jaco(domain, task, seed)
            elif domain in CORR_ENV:
                self._env = make_corr(domain, task, seed)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        self._camera = cameras

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
              -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, (3*len(self._camera),) + self._size , dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        self._steps += 1
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        self._steps = 0
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        img = []
        for camera in self._camera:
            img.append(self._env.physics.render(*self._size, camera_id=camera).transpose(2, 0, 1).copy())
        img = np.concatenate(img, axis=0)
        return img # (6, 64, 64)

class MultiViewJaco:
    def __init__(self, args, name, seed, size=(64, 64), cameras=[0, 1]):
        self.args = args
        domain, task = name.split('-', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            if domain in DMC_ENV:
                from dm_control import suite
                self._env = suite.load(domain, task, task_kwargs={'random':seed})
            elif domain in JACO_ENV:
                self._env = make_jaco(domain, task, seed, self.args.time_limit)
            elif domain in CORR_ENV:
                self._env = make_corr(domain, task, seed)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        self.env_camera1 = MovableCamera(
            physics=self._env.physics,
            height=self._size[0],
            width=self._size[1])
        self.env_camera2 = MovableCamera(
            physics=self._env.physics,
            height=self._size[0],
            width=self._size[1])
        self.init_viewer_cam_lookat1, self.init_viewer_cam_distance1, self.init_viewer_cam_azimuth1, self.init_viewer_cam_elevation1 = np.array([0.0, 0.0, 0.0]), 1.35, 90, -45
        self.init_viewer_cam_lookat2, self.init_viewer_cam_distance2, self.init_viewer_cam_azimuth2, self.init_viewer_cam_elevation2 = np.array([0.0, 0.0, 0.0]), 1.35, 0, -45
        print("Camera1 | lookat: {}, distance: {}, azimuth: {}, elevation: {}".format(self.init_viewer_cam_lookat1, self.init_viewer_cam_distance1, self.init_viewer_cam_azimuth1, self.init_viewer_cam_elevation1))
        print("Camera2 | lookat: {}, distance: {}, azimuth: {}, elevation: {}".format(self.init_viewer_cam_lookat2, self.init_viewer_cam_distance2, self.init_viewer_cam_azimuth2, self.init_viewer_cam_elevation2))

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
          spaces[key] = gym.spaces.Box(
              -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, (3*2,) + self._size , dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        self._steps += 1
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs['image'] = self.render(mode='rgb_array')
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        self._steps = 0
        self.should_reset = True
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render([0, 0, 0])
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
          raise ValueError("Only render mode 'rgb_array' is supported.")
        self.env_camera1 = MovableCamera(
            physics=self._env.physics,
            height=self._size[0],
            width=self._size[1])
        self.env_camera2 = MovableCamera(
            physics=self._env.physics,
            height=self._size[0],
            width=self._size[1])
        self.env_camera1.set_pose(self.init_viewer_cam_lookat1, self.init_viewer_cam_distance1, self.init_viewer_cam_azimuth1, self.init_viewer_cam_elevation1)
        self.env_camera2.set_pose(self.init_viewer_cam_lookat2, self.init_viewer_cam_distance2, self.init_viewer_cam_azimuth2, self.init_viewer_cam_elevation2)
        image1 = self.env_camera1.render(
            overlays=(), depth=False, segmentation=False,
            scene_option=None, render_flag_overrides=None).transpose(2, 0, 1).copy()
        self.env_camera1._scene.free()  # pylint: disable=protected-access
        image2 = self.env_camera2.render(
            overlays=(), depth=False, segmentation=False,
            scene_option=None, render_flag_overrides=None).transpose(2, 0, 1).copy()
        self.env_camera2._scene.free()  # pylint: disable=protected-access
        image = np.concatenate([image1, image2], axis=0)
        return image # (6, 64, 64)

class DeepMindControl:

    def __init__(self, args, name, seed, size=(64, 64), camera=None):
        self.args = args
        domain, task = name.split('-', 1)
        if domain == 'cup':  # Only domain with multiple words.
          domain = 'ball_in_cup'
        if isinstance(domain, str):
            if domain in DMC_ENV:
                from dm_control import suite
                self._env = suite.load(domain, task, task_kwargs={'random':seed})
            elif domain in JACO_ENV:
                self._env = make_jaco(domain, task, seed)
            elif domain in CORR_ENV:
                self._env = make_corr(domain, task, seed)
        else:
          assert task is None
          self._env = domain()
        self._size = size
        if camera is None:
          camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
          spaces[key] = gym.spaces.Box(
              -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, (3,) + self._size , dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        self._steps += 1
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs['image'] = self.render().transpose(2, 0, 1).copy()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        self._steps = 0
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render().transpose(2, 0, 1).copy()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
          raise ValueError("Only render mode 'rgb_array' is supported.")
        if self.args.change_camera_freq > 0:
            return self._env.physics.render(*self._size, camera_id=int(int(self._steps//self.args.change_camera_freq)%2))
        return self._env.physics.render(*self._size, camera_id=self._camera)

class FixedCameraDeepMindControl:
    def __init__(self, args, name, seed, size=(64, 64)):
        self.args = args
        domain, task = name.split('-', 1)
        if domain == 'cup':  # Only domain with multiple words.
          domain = 'ball_in_cup'
        if isinstance(domain, str):
            if domain in DMC_ENV:
                from dm_control import suite
                self._env = suite.load(domain, task, task_kwargs={'random':seed})
            elif domain in JACO_ENV:
                self._env = make_jaco(domain, task, seed)
            elif domain in CORR_ENV:
                self._env = make_corr(domain, task, seed)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        self.env_camera = MovableCamera(
            physics=self._env.physics,
            height=self._size[0],
            width=self._size[1])
        self.viewer_cam_lookat, self.viewer_cam_distance, self.viewer_cam_azimuth, self.viewer_cam_elevation = self.env_camera.get_pose()
        print("lookat: {}, distance: {}, azimuth: {}, elevation: {}".format(self.viewer_cam_lookat, self.viewer_cam_distance, self.viewer_cam_azimuth, self.viewer_cam_elevation))
        self.viewer_cam_lookat = self._env.physics.named.data.geom_xpos[self.args.lookat]
        self.viewer_cam_distance = args.camera_distance
        self.viewer_cam_azimuth = args.camera_azimuth
        self.viewer_cam_elevation = args.camera_elevation
        self.env_camera.set_pose(self.viewer_cam_lookat, self.viewer_cam_distance, self.viewer_cam_azimuth, self.viewer_cam_elevation)
        print('After setting camera pose:')
        print("distance: {}, azimuth: {}, elevation: {}".format(self.viewer_cam_distance, self.viewer_cam_azimuth, self.viewer_cam_elevation))

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
          spaces[key] = gym.spaces.Box(
              -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, (3,) + self._size , dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        self._steps += 1
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs['image'] = self.render(mode='rgb_array').transpose(2, 0, 1).copy()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        self._steps = 0
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render(mode='rgb_array').transpose(2, 0, 1).copy()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
          raise ValueError("Only render mode 'rgb_array' is supported.")
        self.viewer_cam_lookat = self._env.physics.named.data.geom_xpos[self.args.lookat]
        self.env_camera = MovableCamera(
            physics=self._env.physics,
            height=self._size[0],
            width=self._size[1])
        self.env_camera.set_pose(self.viewer_cam_lookat, self.viewer_cam_distance, self.viewer_cam_azimuth, self.viewer_cam_elevation)
        image = self.env_camera.render(
            overlays=(), depth=False, segmentation=False,
            scene_option=None, render_flag_overrides=None)
        self.env_camera._scene.free()  # pylint: disable=protected-access
        return image

class ChangeCameraDeepMindControl:
    def __init__(self, args, name, seed, size=(64, 64)):
        self.args = args
        domain, task = name.split('-', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            if domain in DMC_ENV:
                from dm_control import suite
                self._env = suite.load(domain, task, task_kwargs={'random':seed})
            elif domain in JACO_ENV:
                self._env = make_jaco(domain, task, seed)
            elif domain in CORR_ENV:
                self._env = make_corr(domain, task, seed)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        self.env_camera = MovableCamera(
            physics=self._env.physics,
            height=self._size[0],
            width=self._size[1])
        self.init_viewer_cam_lookat, self.init_viewer_cam_distance, self.init_viewer_cam_azimuth, self.init_viewer_cam_elevation  = self.env_camera.get_pose()
        print("lookat: [{:.3f}, {:.3f}, {:.3f}], distance: {:.3f}, azimuth: {:.3f}, elevation: {:.3f}".format(self.init_viewer_cam_lookat[0], self.init_viewer_cam_lookat[1], self.init_viewer_cam_lookat[2], self.init_viewer_cam_distance, self.init_viewer_cam_azimuth, self.init_viewer_cam_elevation))
        self.viewer_cam_lookat, self.viewer_cam_distance, self.viewer_cam_azimuth, self.viewer_cam_elevation = self.init_viewer_cam_lookat, self.init_viewer_cam_distance, self.init_viewer_cam_azimuth, self.init_viewer_cam_elevation
        self.legal_cam_distance = [0, max(10, self.init_viewer_cam_distance)]
        self.legal_cam_azimuth = [0, 180]
        self.legal_cam_elevation = [-90, 90]
        self.legal_cam_min = np.array([self.legal_cam_distance[0],self.legal_cam_azimuth[0],self.legal_cam_elevation[0]])
        self.legal_cam_max = np.array([self.legal_cam_distance[1],self.legal_cam_azimuth[1],self.legal_cam_elevation[1]])

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
          spaces[key] = gym.spaces.Box(
              -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, (3,) + self._size , dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        self._steps += 1
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs['image'] = self.render(mode='rgb_array').transpose(2, 0, 1).copy()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def normalize_sensor_state(self, sensor_state):
        sensor_state = (sensor_state - self.legal_cam_min) / (self.legal_cam_max - self.legal_cam_min) # normalize into range [0, 1]
        return sensor_state

    def step_camera(self, sensor_action, no_set=True):
        # ----------------------- change camera via sensor action ------------------------------
        if no_set:
            self.viewer_cam_distance = self.env_camera.get_pose().distance + sensor_action[0] * self.args.sensor_action_scale
            self.viewer_cam_azimuth = self.env_camera.get_pose().azimuth + sensor_action[1] * self.args.sensor_action_scale
            self.viewer_cam_elevation = self.env_camera.get_pose().elevation + sensor_action[2] * self.args.sensor_action_scale
        else:
            self.viewer_cam_distance = sensor_action[0]
            self.viewer_cam_azimuth = sensor_action[1]
            self.viewer_cam_elevation = sensor_action[2]
        self.viewer_cam_lookat = self._env.physics.named.data.geom_xpos[self.args.lookat] # self.env_camera.get_pose().lookat + sensor_action[3:]
        self.env_camera.set_pose(self.viewer_cam_lookat, self.viewer_cam_distance, self.viewer_cam_azimuth, self.viewer_cam_elevation)
        notdone_condition = (self.legal_cam_distance[0] <= self.viewer_cam_distance <= self.legal_cam_distance[1]) \
                            and (self.legal_cam_azimuth[0] <= self.viewer_cam_azimuth <= self.legal_cam_azimuth[1]) \
                            and (self.legal_cam_elevation[0] <= self.viewer_cam_elevation <= self.legal_cam_elevation[1])
        print("lookat: [{:.3f}, {:.3f}, {:.3f}], distance: {:.3f}, azimuth: {:.3f}, elevation: {:.3f}, done: {}".format(self.viewer_cam_lookat[0], self.viewer_cam_lookat[1], self.viewer_cam_lookat[2], self.viewer_cam_distance, self.viewer_cam_azimuth, self.viewer_cam_elevation, not notdone_condition))
        sensor_state = np.array([self.viewer_cam_distance, self.viewer_cam_azimuth, self.viewer_cam_elevation])
        sensor_state = self.normalize_sensor_state(sensor_state)
        if notdone_condition:
            return sensor_state, False
        else:
            sensor_state = self.reset_camera()
            return sensor_state, True
            

    def reset_camera(self):
        self.should_reset = True
        self.viewer_cam_lookat, self.viewer_cam_distance, self.viewer_cam_azimuth, self.viewer_cam_elevation = self.init_viewer_cam_lookat, self.init_viewer_cam_distance, self.init_viewer_cam_azimuth, self.init_viewer_cam_elevation
        sensor_state = np.array([self.init_viewer_cam_distance, self.init_viewer_cam_azimuth, self.init_viewer_cam_elevation])
        sensor_state = self.normalize_sensor_state(sensor_state)
        return sensor_state

    def reset(self):
        self._steps = 0
        self.reset_camera() # -> True
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render([0, 0, 0]).transpose(2, 0, 1).copy()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
          raise ValueError("Only render mode 'rgb_array' is supported.")
        self.env_camera = MovableCamera(
            physics=self._env.physics,
            height=self._size[0],
            width=self._size[1])
        if self.should_reset:
            self.env_camera.set_pose(self.init_viewer_cam_lookat, self.init_viewer_cam_distance, self.init_viewer_cam_azimuth, self.init_viewer_cam_elevation)
            self.should_reset = False
        else:
            self.viewer_cam_lookat = self._env.physics.named.data.geom_xpos[self.args.lookat]
            self.env_camera.set_pose(self.viewer_cam_lookat, self.viewer_cam_distance, self.viewer_cam_azimuth, self.viewer_cam_elevation)
        image = self.env_camera.render(
            overlays=(), depth=False, segmentation=False,
            scene_option=None, render_flag_overrides=None)
        self.env_camera._scene.free()  # pylint: disable=protected-access
        return image


class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
          done = True
          if 'discount' not in info:
            info['discount'] = np.array(1.0).astype(np.float32)
          self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class ActionRepeat:

    def __init__(self, env, amount):
        self._env = env
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
          obs, reward, done, info = self._env.step(action)
          total_reward += reward
          current_step += 1
        return obs, total_reward, done, info

class NormalizeActions:

    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class ObsDict:

    def __init__(self, env, key='obs'):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = {self._key: self._env.observation_space}
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {self._key: np.array(obs)}
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = {self._key: np.array(obs)}
        return obs


class OneHotAction:

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
          raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert 'reward' not in spaces
        spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs['reward'] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['reward'] = 0.0
        return obs


class ResizeImage:

    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size
        self._keys = [
            k for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
          from PIL import Image
          self._Image = Image

    def __getattr__(self, name):
        if name.startswith('__'):
          raise AttributeError(name)
        try:
          return getattr(self._env, name)
        except AttributeError:
          raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
          shape = self._size + spaces[key].shape[2:]
          spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
          obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
          obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage:

    def __init__(self, env, key='image'):
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name):
        if name.startswith('__'):
          raise AttributeError(name)
        try:
          return getattr(self._env, name)
        except AttributeError:
          raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        obs[self._key] = self._env.render('rgb_array')
        return obs

    def reset(self):
        obs = self._env.reset()
        obs[self._key] = self._env.render('rgb_array')
        return obs


# -------------------------------- jaco ----------------------------------------------
"""A task where the goal is to move the hand close to a target prop or site."""

import collections

from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.manipulation.shared import arenas
from dm_control.manipulation.shared import cameras
from dm_control.manipulation.shared import constants
from dm_control.manipulation.shared import observations
from dm_control.manipulation.shared import registry
from dm_control.manipulation.shared import robots
from dm_control.manipulation.shared import tags
from dm_control.manipulation.shared import workspaces
from dm_control.utils import rewards

_ReachWorkspace = collections.namedtuple(
    '_ReachWorkspace', ['target_bbox', 'tcp_bbox', 'arm_offset'])

# Ensures that the props are not touching the table before settling.
_PROP_Z_OFFSET = 0.001

_DUPLO_WORKSPACE = _ReachWorkspace(
    target_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _PROP_Z_OFFSET),
        upper=(0.1, 0.1, _PROP_Z_OFFSET)),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, 0.2),
        upper=(0.1, 0.1, 0.4)),
    arm_offset=robots.ARM_OFFSET)

_SITE_WORKSPACE = _ReachWorkspace(
    target_bbox=workspaces.BoundingBox(
        lower=(-0.2, -0.2, 0.02),
        upper=(0.2, 0.2, 0.4)),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.2, -0.2, 0.02),
        upper=(0.2, 0.2, 0.4)),
    arm_offset=robots.ARM_OFFSET)

_TARGET_RADIUS = 0.05
_TIME_LIMIT = 10.

TASKS = {
    'reach_top_left':  workspaces.BoundingBox(
        lower=(-0.09, 0.09, _PROP_Z_OFFSET),
        upper=(-0.09, 0.09, _PROP_Z_OFFSET)),
    'reach_top_right': workspaces.BoundingBox(
        lower=(0.09, 0.09, _PROP_Z_OFFSET),
        upper=(0.09, 0.09, _PROP_Z_OFFSET)),
    'reach_bottom_left': workspaces.BoundingBox(
        lower=(-0.09, -0.09, _PROP_Z_OFFSET),
        upper=(-0.09, -0.09, _PROP_Z_OFFSET)),
    'reach_bottom_right': workspaces.BoundingBox(
        lower=(0.09, -0.09, _PROP_Z_OFFSET),
        upper=(0.09, -0.09, _PROP_Z_OFFSET)),
}


def make_jaco(domain, task_name, seed, time_limit=10):
  obs_type = 'pixels'
  obs_settings = observations.VISION if obs_type == 'pixels' else observations.PERFECT_FEATURES
  task = _reach(domain+'_'+task_name, obs_settings=obs_settings, use_site=False)
  return composer.Environment(task, time_limit=time_limit, random_state=seed)



class MTReach(composer.Task):
  """Bring the hand close to a target prop or site."""

  def __init__(
      self, task_id, arena, arm, hand, prop, obs_settings, workspace, control_timestep):
    """Initializes a new `Reach` task.

    Args:
      arena: `composer.Entity` instance.
      arm: `robot_base.RobotArm` instance.
      hand: `robot_base.RobotHand` instance.
      prop: `composer.Entity` instance specifying the prop to reach to, or None
        in which case the target is a fixed site whose position is specified by
        the workspace.
      obs_settings: `observations.ObservationSettings` instance.
      workspace: `_ReachWorkspace` specifying the placement of the prop and TCP.
      control_timestep: Float specifying the control timestep in seconds.
    """
    self._arena = arena
    self._arm = arm
    self._hand = hand
    self._arm.attach(self._hand)
    self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
    self.control_timestep = control_timestep
    self._tcp_initializer = initializers.ToolCenterPointInitializer(
        self._hand, self._arm,
        position=distributions.Uniform(*workspace.tcp_bbox),
        quaternion=workspaces.DOWN_QUATERNION)

    # Add custom camera observable.
    self._task_observables = cameras.add_camera_observables(
        arena, obs_settings, cameras.FRONT_CLOSE)

    target_pos_distribution = distributions.Uniform(*TASKS[task_id])
    self._prop = prop
    if prop:
      # The prop itself is used to visualize the target location.
      self._make_target_site(parent_entity=prop, visible=False)
      self._target = self._arena.add_free_entity(prop)
      self._prop_placer = initializers.PropPlacer(
          props=[prop],
          position=target_pos_distribution,
          quaternion=workspaces.uniform_z_rotation,
          settle_physics=True)
    else:
      self._target = self._make_target_site(parent_entity=arena, visible=True)
      self._target_placer = target_pos_distribution

      obs = observable.MJCFFeature('pos', self._target)
      obs.configure(**obs_settings.prop_pose._asdict())
      self._task_observables['target_position'] = obs

    # Add sites for visualizing the prop and target bounding boxes.
    workspaces.add_bbox_site(
        body=self.root_entity.mjcf_model.worldbody,
        lower=workspace.tcp_bbox.lower, upper=workspace.tcp_bbox.upper,
        rgba=constants.GREEN, name='tcp_spawn_area')
    workspaces.add_bbox_site(
        body=self.root_entity.mjcf_model.worldbody,
        lower=workspace.target_bbox.lower, upper=workspace.target_bbox.upper,
        rgba=constants.BLUE, name='target_spawn_area')

  def _make_target_site(self, parent_entity, visible):
    return workspaces.add_target_site(
        body=parent_entity.mjcf_model.worldbody,
        radius=_TARGET_RADIUS, visible=visible,
        rgba=constants.RED, name='target_site')

  @property
  def root_entity(self):
    return self._arena

  @property
  def arm(self):
    return self._arm

  @property
  def hand(self):
    return self._hand

  @property
  def task_observables(self):
    return self._task_observables

  def get_reward(self, physics):
    hand_pos = physics.bind(self._hand.tool_center_point).xpos
    target_pos = physics.bind(self._target).xpos
    distance = np.linalg.norm(hand_pos - target_pos)
    return rewards.tolerance(
        distance, bounds=(0, _TARGET_RADIUS), margin=_TARGET_RADIUS)

  def initialize_episode(self, physics, random_state):
    self._hand.set_grasp(physics, close_factors=random_state.uniform())
    self._tcp_initializer(physics, random_state)
    if self._prop:
      self._prop_placer(physics, random_state)
    else:
      physics.bind(self._target).pos = (
          self._target_placer(random_state=random_state))


def _reach(task_id, obs_settings, use_site):
    """Configure and instantiate a `Reach` task.

    Args:
        obs_settings: An `observations.ObservationSettings` instance.
        use_site: Boolean, if True then the target will be a fixed site, otherwise
        it will be a moveable Duplo brick.

    Returns:
        An instance of `reach.Reach`.
    """
    arena = arenas.Standard()
    arm = robots.make_arm(obs_settings=obs_settings)
    hand = robots.make_hand(obs_settings=obs_settings)
    if use_site:
        workspace = _SITE_WORKSPACE
        prop = None
    else:
        workspace = _DUPLO_WORKSPACE
        prop = props.Duplo(observable_options=observations.make_options(
            obs_settings, observations.FREEPROP_OBSERVABLES))
    task = MTReach(task_id, arena=arena, arm=arm, hand=hand, prop=prop,
                obs_settings=obs_settings,
                workspace=workspace,
                control_timestep=constants.CONTROL_TIMESTEP)
    return task