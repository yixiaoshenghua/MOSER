import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
from sklearn.decomposition import IncrementalPCA
from replay_buffer import ReplayBuffer, SensorBuffer
from models import RSSM, ConvEncoder, ConvDecoder, DenseDecoder, ActionDecoder, DiagGaussianActor, DoubleQCritic, RandomActor, ConstantActor
from utils import *


class MOSER:

    def __init__(self, args, env, obs_shape, action_size, device, restore=False):

        self.args = args
        self.env = env
        if self.args.actor_grad == 'auto':
            self.args.actor_grad = 'dynamics' if self.args.algo == 'MOSERv1' else 'reinforce'
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.device = device
        self.restore = args.restore
        self.restore_path = args.checkpoint_path
        self.data_buffer = ReplayBuffer(self.args.buffer_size, self.obs_shape, self.action_size,
                                                    self.args.train_seq_len, self.args.batch_size)
        if self.args.combine_offline_datadir is not None:
            self.data_buffer.add_from_files(self.args.combine_offline_datadir)
        self.step = args.seed_steps
        self._build_model(restore=self.restore)

    def _build_model(self, restore):
        # ---------------------------- build soft-actor-critic ---------------------------------------
        if self.args.sensor_mode == 'random':
            self.sensor_actor = RandomActor(
                                low=torch.tensor(self.env.legal_cam_min).float().to(self.device),
                                high=torch.tensor(self.env.legal_cam_max).float().to(self.device)).to(self.device)
        elif self.args.sensor_mode == 'constant':
            self.sensor_actor = ConstantActor(
                                value=torch.tensor([2, 10, 10]).to(self.device)).to(self.device)
        else:
            self.sensor_actor = DiagGaussianActor(
                            obs_dim = self.args.sensor_state_size,
                            action_dim = self.args.sensor_action_size,
                            hidden_dim = 256,
                            hidden_depth = 2,
                            log_std_bounds = [-2, 5]).to(self.device)
            self.sensor_critic = DoubleQCritic(
                                obs_dim = self.args.sensor_state_size,
                                action_dim = self.args.sensor_action_size,
                                hidden_dim = 256,
                                hidden_depth = 2).to(self.device)
            self.sensor_critic_target = DoubleQCritic(
                                obs_dim = self.args.sensor_state_size,
                                action_dim = self.args.sensor_action_size,
                                hidden_dim = 256,
                                hidden_depth = 2).to(self.device)
            self.sensor_critic_target.load_state_dict(self.sensor_critic.state_dict())
            self.sensor_log_alpha = torch.tensor(np.log(self.args.init_temperature)).to(self.device)
            self.sensor_log_alpha.requires_grad = True
        self.sensor_target_entropy = -self.args.sensor_action_size
        self.sensor_buffer = SensorBuffer((self.args.sensor_state_size,), (self.args.sensor_action_size,), int(self.args.total_steps/(self.args.action_repeat*self.args.sensor_action_frequency)), self.device)
        # --------------------------------------------------------------------------------------------
        self.rssm = RSSM(
                    action_size = self.action_size if self.transform_action_size == 0 else self.transform_action_size,
                    stoch_size = self.args.stoch_size,
                    deter_size = self.args.deter_size,
                    hidden_size = self.args.hidden_size,
                    obs_embed_size = self.args.obs_embed_size,
                    activation = self.args.dense_activation_function,
                    ensemble = self.args.ensemble,
                    discrete = self.args.discrete,
                    future=self.args.rssm_attention,
                    reverse=self.args.rssm_reverse,
                    device=self.device).to(self.device)

        self.motor_actor = ActionDecoder(
                     action_size = self.action_size,
                     stoch_size = self.args.stoch_size,
                     deter_size = self.args.deter_size,
                     units = self.args.num_units,
                     n_layers = 4,
                     dist = self.args.actor_dist,
                     min_std = self.args.actor_min_std,
                     init_std  = self.args.actor_init_std,
                     activation = self.args.dense_activation_function,
                     discrete = self.args.discrete).to(self.device)
        
        self.obs_encoder  = ConvEncoder(
                            input_shape= self.obs_shape,
                            embed_size = self.args.obs_embed_size,
                            activation =self.args.cnn_activation_function).to(self.device)
        self.obs_decoder  = ConvDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size + self.args.sensor_action_size,
                            output_shape=self.obs_shape,
                            activation = self.args.cnn_activation_function,
                            discrete=self.args.discrete).to(self.device)
        self.reward_model = DenseDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (1,),
                            n_layers = 2,
                            units=self.args.num_units,
                            activation= self.args.dense_activation_function,
                            dist = 'normal',
                            discrete = self.args.discrete).to(self.device)
        self.motor_critic  = DenseDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (1,),
                            n_layers = 3,
                            units = self.args.num_units,
                            activation= self.args.dense_activation_function,
                            dist = 'normal',
                            discrete = self.args.discrete).to(self.device) 
         
        if self.args.slow_target:
            self.target_critic = DenseDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (1,),
                            n_layers = 3,
                            units = self.args.num_units,
                            activation= self.args.dense_activation_function,
                            dist = 'normal',
                            discrete = self.args.discrete).to(self.device) 
            self._updates = 0
        if self.args.use_disc_model:  
            self.discount_model = DenseDecoder(
                                stoch_size = self.args.stoch_size,
                                deter_size = self.args.deter_size,
                                output_shape = (1,),
                                n_layers = 2,
                                units=self.args.num_units,
                                activation= self.args.dense_activation_function,
                                dist = 'binary',
                                discrete = self.args.discrete).to(self.device)
        
        if self.args.use_disc_model:
            self.world_model_params = list(self.rssm.parameters()) + list(self.obs_decoder.parameters()) \
                + list(self.reward_model.parameters()) + list(self.discount_model.parameters()) + list(self.obs_encoder.parameters())
        else:
            self.world_model_params = list(self.rssm.parameters()) + list(self.obs_decoder.parameters()) \
                + list(self.reward_model.parameters()) + list(self.obs_encoder.parameters())
    
        self.world_model_opt = optim.Adam(self.world_model_params, self.args.model_learning_rate)
        self.motor_critic_opt = optim.Adam(self.motor_critic.parameters(), self.args.value_learning_rate)
        self.motor_actor_opt = optim.Adam(self.motor_actor.parameters(), self.args.actor_learning_rate)
        # ---------------------------- build soft-actor-critic ---------------------------------------
        if self.args.sensor_mode == 'train':
            self.sensor_actor_opt = optim.Adam(self.sensor_actor.parameters(), lr=self.args.sensor_actor_lr, betas=[0.9, 0.999])
            self.sensor_critic_opt = torch.optim.Adam(self.sensor_critic.parameters(), lr=self.args.sensor_critic_lr, betas=[0.9, 0.999])
            self.sensor_log_alpha_opt = torch.optim.Adam([self.sensor_log_alpha], lr=self.args.sensor_alpha_lr, betas=[0.9, 0.999])
        # --------------------------------------------------------------------------------------------
        if self.args.use_disc_model:
            self.world_model_modules = [self.rssm, self.obs_encoder, self.obs_decoder, self.reward_model, self.discount_model]
        else:
            self.world_model_modules = [self.rssm, self.obs_encoder, self.obs_decoder, self.reward_model]
        self.value_modules = [self.motor_critic]
        self.motor_actor_modules = [self.motor_actor]

        if restore:
            self.load(self.restore_path)

    def motor_actor_loss(self):
        loss_dict, log_dict = {}, {}
        with torch.no_grad():
            posterior = self.rssm.detach_state(self.rssm.seq_to_batch(self.posterior))

        with FreezeParameters(self.world_model_modules):
            imag_states, imag_actions, imag_feats = self.imagine(posterior, self.args.imagine_horizon)

        self.imag_feat = self.rssm.get_feat(imag_states)

        with FreezeParameters(self.world_model_modules + self.value_modules):
            imag_rew_dist = self.reward_model(self.imag_feat)
            imag_val_dist = self.motor_critic(self.imag_feat)

            imag_rews = imag_rew_dist.mean
            imag_vals = imag_val_dist.mean
            if self.args.use_disc_model:
                imag_disc_dist = self.discount_model(self.imag_feat)
                discounts = imag_disc_dist.mean().detach()
            else:
                discounts =  self.args.discount * torch.ones_like(imag_rews).detach()

        self.returns = compute_return(imag_rews[:-1], imag_vals[:-1],discounts[:-1] \
                                         ,self.args.td_lambda, imag_vals[-1])

        discounts = torch.cat([torch.ones_like(discounts[:1]), discounts[1:-1]], 0)
        self.discounts = torch.cumprod(discounts, 0).detach()
        actor_loss = -torch.mean(self.discounts * self.returns)
        
        loss_dict['actor_loss'] = actor_loss
        log_dict['train/actor_loss'] = actor_loss.item()
        
        return loss_dict, log_dict

    def motor_critic_loss(self):
        loss_dict, log_dict = {}, {}
        with torch.no_grad():
            value_feat = self.imag_feat[:-1].detach()
            discount   = self.discounts.detach()
            value_targ = self.returns.detach()

        value_dist = self.motor_critic(value_feat)  
        value_loss = -torch.mean(self.discounts * value_dist.log_prob(value_targ).unsqueeze(-1))

        loss_dict['value_loss'] = value_loss
        log_dict['train/value_loss'] = value_loss.item()

        return loss_dict, log_dict

    def world_model_loss(self, obs, acs, rews, nonterms, sensor_acs):
        loss_dict, log_dict = {}, {}
        obs = preprocess_obs(obs)
        obs_embed = self.obs_encoder(obs[1:]) # (T-1, n, e)
        #-----------------------------------------------------------------

        init_state = self.rssm.init_state(self.args.batch_size, self.device)
        prior, self.posterior = self.rssm.observe_rollout(obs_embed, acs[:-1], nonterms[:-1], init_state, self.args.train_seq_len-1)
        features = self.rssm.get_feat(self.posterior)
        rew_dist = self.reward_model(features)
        obs_dist = self.obs_decoder(torch.cat([features, sensor_acs[:-1]], dim=-1))
        if self.args.use_disc_model:
            disc_dist = self.discount_model(features)

        prior_dist = self.rssm.get_dist(prior)
        prior_ent = torch.mean(prior_dist.entropy())
        post_dist = self.rssm.get_dist(self.posterior)
        post_ent = torch.mean(post_dist.entropy())
        if self.args.kl_balancing:
            post_no_grad = self.rssm.detach_state(self.posterior)
            prior_no_grad = self.rssm.detach_state(prior)
            
            # kl_balancing
            kl_loss = self.args.kl_alpha * (torch.mean(distributions.kl.kl_divergence(
                               self.rssm.get_dist(post_no_grad), prior_dist)))
            kl_loss += (1-self.args.kl_alpha) * (torch.mean(distributions.kl.kl_divergence(
                               post_dist, self.rssm.get_dist(prior_no_grad))))
        else:
            kl_loss = torch.mean(distributions.kl.kl_divergence(post_dist, prior_dist))
            kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), self.args.free_nats))

        obs_loss = -torch.mean(obs_dist.log_prob(obs[1:])) 
        rew_loss = -torch.mean(rew_dist.log_prob(rews[:-1]))
        if self.args.use_disc_model:
            disc_loss = -torch.mean(disc_dist.log_prob(nonterms[:-1]))
            loss_dict['disc_loss'] = disc_loss
            log_dict['disc_loss'] = disc_loss.item()

        if self.args.use_disc_model:
            model_loss = self.args.kl_loss_coeff * kl_loss + obs_loss + rew_loss + self.args.disc_loss_coeff * disc_loss
        else:
            model_loss = self.args.kl_loss_coeff * kl_loss + obs_loss + rew_loss 
        
        loss_dict['kl_loss'] = kl_loss
        loss_dict['obs_loss'] = obs_loss
        loss_dict['rew_loss'] = rew_loss
        loss_dict['model_loss'] = model_loss
        log_dict['world_model/prior_ent'] = prior_ent.item()
        log_dict['world_model/post_ent'] = post_ent.item()
        log_dict['world_model/mi_gain'] = prior_ent.item() - post_ent.item()
        log_dict['world_model/post_std'] = torch.mean(self.posterior['std']).item() if 'std' in self.posterior.keys() else 0
        log_dict['world_model/kl_loss'] = kl_loss.item()
        log_dict['world_model/obs_loss'] = obs_loss.item()
        log_dict['world_model/rew_loss'] = rew_loss.item()
        log_dict['world_model/model_loss'] = model_loss.item()

        return loss_dict, log_dict

    def update_world_model(self, obs, acs, rews, nonterms, sensor_acs):
        wm_loss_dict, wm_log_dict = self.world_model_loss(obs, acs, rews, nonterms, sensor_acs)
        model_loss = wm_loss_dict['model_loss']
        self.world_model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model_params, self.args.grad_clip_norm)
        self.world_model_opt.step()
        return wm_log_dict
    
    def update_motor_actor(self):
        ac_loss_dict, ac_log_dict = self.motor_actor_loss()
        actor_loss = ac_loss_dict['actor_loss']
        self.motor_actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.motor_actor.parameters(), self.args.grad_clip_norm)
        self.motor_actor_opt.step()
        return ac_log_dict
    
    def update_motor_critic(self):
        val_loss_dict, val_log_dict = self.motor_critic_loss()
        value_loss = val_loss_dict['value_loss']
        self.motor_critic_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.motor_critic.parameters(), self.args.grad_clip_norm)
        self.motor_critic_opt.step()
        return val_log_dict

    def target(self, feat, reward, disc):
        if self.args.slow_target:
            value = self.target_critic(feat).mean
        else:
            value = self.motor_critic(feat).mean
        target = compute_return(reward[:-1], value[:-1], disc[:-1], \
                                self.args.td_lambda, value[-1])
        weight = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[1:-1]], 0).detach(), 0)
        return target, weight

    def update_slow_target(self):
        if self.args.slow_target:
            if self._updates % self.args.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.args.slow_target_fraction)
                for s, d in zip(self.motor_critic.parameters(), self.target_critic.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

    def update_sensor_policy(self):
        def update_critic(obs, action, reward, next_obs, not_done):
            log_dict = {}
            dist = self.sensor_actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.sensor_critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                target_Q2) - self.sensor_log_alpha.exp().detach() * log_prob
            target_Q = reward + (not_done * self.args.discount * target_V)
            target_Q = target_Q.detach()

            # get current Q estimates
            current_Q1, current_Q2 = self.sensor_critic(obs, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q)
            log_dict['train/sensor_critic_loss'] = critic_loss.item()

            # Optimize the critic
            self.sensor_critic_opt.zero_grad()
            critic_loss.backward()
            self.sensor_critic_opt.step()
            return log_dict

        def update_actor_and_alpha(obs):
            log_dict = {}
            dist = self.sensor_actor(obs)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            actor_Q1, actor_Q2 = self.sensor_critic(obs, action)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.sensor_log_alpha.exp().detach() * log_prob - actor_Q).mean()

            log_dict['train/sensor_actor_loss'] = actor_loss.item()
            log_dict['train/sensor_actor_target_entropy'] = self.sensor_target_entropy
            log_dict['train/sensor_actor_entropy'] = -log_prob.mean().item()

            # optimize the actor
            self.sensor_actor_opt.zero_grad()
            actor_loss.backward()
            self.sensor_actor_opt.step()

            self.sensor_log_alpha_opt.zero_grad()
            alpha_loss = (self.sensor_log_alpha.exp() *
                        (-log_prob - self.sensor_target_entropy).detach()).mean()
            log_dict['train/sensor_alpha_loss'] = alpha_loss.item()
            log_dict['train/sensor_alpha_value'] = self.sensor_log_alpha.exp().item()
            alpha_loss.backward()
            self.sensor_log_alpha_opt.step()
            return log_dict
        
        log_dict = {}
        obs, action, reward, next_obs, nonterm = self.sensor_buffer.sample(self.args.sensor_batch_size)

        log_dict['train/sensor_batch_reward'] = reward.mean().item()

        log_dict.update(update_critic(obs, action, reward, next_obs, nonterm))

        if self.step % self.args.sensor_actor_update_frequency == 0:
            log_dict.update(update_actor_and_alpha(obs))

        if self.step % self.args.sensor_critic_target_update_frequency == 0:
            soft_update_params(self.sensor_critic, self.sensor_critic_target,
                                     self.args.sensor_critic_tau)
        return log_dict

    def update(self):
        log_dict = {}
        obs, acs, rews, terms, sensor_acs = self.data_buffer.sample()
        obs  = torch.tensor(obs, dtype=torch.float32).to(self.device)
        acs  = torch.tensor(acs, dtype=torch.float32).to(self.device)
        sensor_acs = torch.tensor(sensor_acs, dtype=torch.float32).to(self.device)
        rews = torch.tensor(rews, dtype=torch.float32).to(self.device).unsqueeze(-1)
        nonterms = torch.tensor((1.0-terms), dtype=torch.float32).to(self.device).unsqueeze(-1)

        wm_log_dict = self.update_world_model(obs, acs, rews, nonterms, sensor_acs)
        ac_log_dict = self.update_motor_actor()
        val_log_dict = self.update_motor_critic()
        self.update_slow_target()

        log_dict.update(wm_log_dict)
        log_dict.update(ac_log_dict)
        log_dict.update(val_log_dict)

        return log_dict

    def act_with_world_model(self, obs, prev_state, prev_action, explore=False):
        obs = obs['image']
        obs  = torch.tensor(obs.copy(), dtype=torch.float32).to(self.device).unsqueeze(0)
        obs = preprocess_obs(obs)
        obs_embed = self.obs_encoder(obs)
        
        prior, posterior = self.rssm.observe_step(prev_state, prev_action, obs_embed)
        features = self.rssm.get_feat(posterior)
        action = self.motor_actor(features, deter=not explore) 
        if explore:
            action = self.motor_actor.add_exploration(action, self.args.action_noise)

        return  prior, posterior, action

    def act_and_collect_data(self, env, collect_steps):
        log_dict = AvgDict()
        cam_log_dict = AvgDict()
        obs = env.reset()
        sensor_state = env.reset_camera()
        done = False
        prev_state = self.rssm.init_state(1, self.device)
        prev_action = torch.zeros(1, self.action_size).to(self.device)

        episode_rewards = [0.0]

        for i in range(collect_steps):
            self.step += self.args.action_repeat
            # TODO
            # ------------------------------------ sample sensor action and step the camera -------------------------------------------------------------
            if i % self.args.sensor_action_frequency == 0:
                priors, posteriors = [], []
                sensor_state = torch.tensor(sensor_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                sensor_action = self.sensor_actor.sample_action(sensor_state)
                sensor_action = sensor_action.detach().cpu().numpy().squeeze(0)
                next_sensor_state, sensor_done = env.step_camera(sensor_action, no_set=self.args.sensor_mode=='train')
                cam_log_dict.record({
                    'env/lookat_0': env.viewer_cam_lookat[0],
                    'env/lookat_1': env.viewer_cam_lookat[1],
                    'env/lookat_2': env.viewer_cam_lookat[2],
                    'env/distance': env.viewer_cam_distance,
                    'env/azimuth': env.viewer_cam_azimuth,
                    'env/elevation': env.viewer_cam_elevation})
            # -------------------------------------------------------------------------------------------------------------------------------------------
            with torch.no_grad():
                prior, posterior, action = self.act_with_world_model(obs, prev_state, prev_action, explore=True)
                priors.append(prior)
                posteriors.append(posterior)
            action = action[0].cpu().numpy()
            next_obs, rew, done, _ = env.step(action)
            self.data_buffer.add(obs, action, rew, done, sensor_action)

            episode_rewards[-1] += rew

            if done:
                obs = env.reset()
                done = False
                prev_state = self.rssm.init_state(1, self.device)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
                if i!= collect_steps-1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs 
                prev_state = posterior
                prev_action = torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)
        
            # TODO
            if (i+1) % self.args.sensor_action_frequency == 0 and self.args.sensor_mode == 'train':
                # compute the intrinsic reward for sensor policy
                priors, posteriors = self.rssm.stack_states(priors), self.rssm.stack_states(posteriors)
                intrinsic_rew_log_dict = self.get_intrinsic_reward(priors, posteriors, self.args.sensor_action_frequency)
                sensor_reward = intrinsic_rew_log_dict['train/sensor_mi_gain'] * self.args.sensor_reward_mi_gain_coeff \
                                + intrinsic_rew_log_dict['train/sensor_reward_pred'] * self.args.sensor_reward_rew_pred_coeff \
                                + intrinsic_rew_log_dict['train/sensor_obs_like'] * self.args.sensor_reward_obs_like_coeff
                self.sensor_buffer.add(sensor_state.squeeze(0).detach().cpu().numpy(), sensor_action, sensor_reward, next_sensor_state, sensor_done)
                sensor_state = next_sensor_state
                log_dict.update(intrinsic_rew_log_dict)
                if self.sensor_buffer.idx > self.args.sensor_batch_size or self.sensor_buffer.full:
                    for _ in range(self.args.sensor_policy_update_num):
                        train_sensor_log_dict = self.update_sensor_policy()
                        log_dict.record(train_sensor_log_dict)

        log_dict.update(cam_log_dict.dict)
        return np.array(episode_rewards), log_dict
    
    @torch.no_grad()
    def get_intrinsic_reward(self, priors, posteriors, collect_steps):
        log_dict = {}
        # TODO
        # retrieve the data from buffer
        T, B = collect_steps, 1
        retrieve_idxs = np.arange(self.data_buffer.idx + self.data_buffer.size - collect_steps, self.data_buffer.idx) if self.data_buffer.full \
            else np.arange(self.data_buffer.idx - collect_steps, self.data_buffer.idx)
        obs = torch.tensor(self.data_buffer.observations[retrieve_idxs], dtype=torch.float32).reshape((T, B, *self.data_buffer.obs_shape)).to(self.device)
        sensor_acs = torch.tensor(self.data_buffer.sensor_actions[retrieve_idxs], dtype=torch.float32).reshape((T, B, -1)).to(self.device)

        # compute the intrinsic reward for sensor policy
        obs = preprocess_obs(obs)
        features = self.rssm.get_feat(posteriors)
        
        prior_dist = self.rssm.get_dist(priors)
        prior_ent = torch.mean(prior_dist.entropy())
        post_dist = self.rssm.get_dist(posteriors)
        post_ent = torch.mean(post_dist.entropy())
        mi_gain = torch.mean(prior_ent - post_ent)
        log_dict['train/sensor_mi_gain'] = mi_gain.item()

        rew_dist = self.reward_model(features)
        rew_pred = torch.mean(rew_dist.mean)
        log_dict['train/sensor_reward_pred'] = rew_pred.item()

        obs_dist = self.obs_decoder(torch.cat([features, sensor_acs], dim=-1))
        obs_like = torch.mean(obs_dist.log_prob(obs))
        log_dict['train/sensor_obs_like'] = obs_like.item()
        
        return log_dict

    def imagine(self, prev_state, horizon):

        rssm_state = prev_state
        next_states = []
        features = []
        actions = []

        for t in range(horizon):
            feature = self.rssm.get_feat(rssm_state)
            action = self.motor_actor(feature.detach())
            rssm_state = self.rssm.imagine_step(rssm_state, action)
            next_states.append(rssm_state)
            actions.append(action)
            features.append(feature)

        next_states = self.rssm.stack_states(next_states)
        features = torch.cat(features, dim=0)
        actions = torch.cat(actions, dim=0)

        return next_states, actions, features

    def evaluate(self, env, eval_episodes, render=False):

        episode_rew = np.zeros((eval_episodes))

        video_images = [[] for _ in range(eval_episodes)]

        # ---------------- select sensor action via sensor policy
        sensor_state = torch.tensor([[env.viewer_cam_distance, env.viewer_cam_azimuth, env.viewer_cam_elevation]], dtype=torch.float32).to(self.device)
        sensor_action = self.sensor_actor.select_action(sensor_state)
        sensor_action = sensor_action.detach().cpu().numpy().squeeze(0)
        env.step_camera(sensor_action, no_set=self.args.sensor_mode=='train')
        log_dict = {
                    'eval/env_cam_lookat_0': env.viewer_cam_lookat[0],
                    'eval/env_cam_lookat_1': env.viewer_cam_lookat[1],
                    'eval/env_cam_lookat_2': env.viewer_cam_lookat[2],
                    'eval/env_cam_distance': env.viewer_cam_distance,
                    'eval/env_cam_azimuth': env.viewer_cam_azimuth,
                    'eval/env_cam_elevation': env.viewer_cam_elevation,
                }
        for i in range(eval_episodes):
            obs = env.reset()
            done = False
            prev_state = self.rssm.init_state(1, self.device)
            prev_action = torch.zeros(1, self.action_size).to(self.device)

            while not done:
                with torch.no_grad():
                    prior, posterior, action = self.act_with_world_model(obs, prev_state, prev_action)
                action = action[0].cpu().numpy()
                next_obs, rew, done, _ = env.step(action)
                prev_state = posterior
                prev_action = torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)

                episode_rew[i] += rew

                if render:
                    video_images[i].append(obs['image'].transpose(1, 2, 0).copy())
                obs = next_obs

        # video prediction
        obs, acs, rews, terms, sensor_acs = self.data_buffer.sample()
        obs  = torch.tensor(obs, dtype=torch.float32).to(self.device)
        acs  = torch.tensor(acs, dtype=torch.float32).to(self.device)
        nonterms = torch.tensor((1.0-terms), dtype=torch.float32).to(self.device).unsqueeze(-1)
        sensor_acs = torch.tensor(sensor_acs, dtype=torch.float32).to(self.device)
        pred_videos = self.video_pred(obs, acs, nonterms, sensor_acs)

        return log_dict, episode_rew, np.array(video_images[:self.args.max_videos_to_save]), pred_videos # (T, H, W, C)

    def collect_random_episodes(self, env, seed_steps):

        obs = env.reset()
        done = False
        seed_episode_rews = [0.0]

        for i in range(seed_steps):
            action = env.action_space.sample()
            next_obs, rew, done, _ = env.step(action)
            
            self.data_buffer.add(obs, action, rew, done, np.array([0, 0, 0]))
            seed_episode_rews[-1] += rew
            if done:
                obs = env.reset()
                if i!= seed_steps-1:
                    seed_episode_rews.append(0.0)
                done=False  
            else:
                obs = next_obs

        return np.array(seed_episode_rews)

    def save(self, save_path):
        save_dict = {'rssm' : self.rssm.state_dict(),
            'actor': self.motor_actor.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'obs_encoder': self.obs_encoder.state_dict(),
            'obs_decoder': self.obs_decoder.state_dict(),
            'sensor_actor': self.sensor_actor.state_dict(),
            'discount_model': self.discount_model.state_dict() if self.args.use_disc_model else None,
            'motor_actor_optimizer': self.motor_actor_opt.state_dict(),
            'motor_critic_optimizer': self.motor_critic_opt.state_dict(),
            'world_model_optimizer': self.world_model_opt.state_dict(),}
        if self.args.sensor_mode == 'train':
            save_dict.update({'sensor_critic': self.sensor_critic.state_dict(),
            'sensor_critic_target': self.sensor_critic_target.state_dict(),
            'sensor_log_alpha': self.sensor_log_alpha,
            'sensor_actor_optimizer': self.sensor_actor_opt.state_dict(),
            'sensor_critic_optimizer': self.sensor_critic_opt.state_dict(),
            'sensor_log_alpha_optimizer': self.sensor_log_alpha_opt.state_dict(),})
        torch.save(save_dict, save_path)

    def load(self, ckpt_path):

        checkpoint = torch.load(ckpt_path)
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.sensor_actor.load_state_dict(checkpoint['sensor_actor'])
        self.motor_actor.load_state_dict(checkpoint['actor'])
        self.reward_model.load_state_dict(checkpoint['reward_model'])
        self.obs_encoder.load_state_dict(checkpoint['obs_encoder'])
        self.obs_decoder.load_state_dict(checkpoint['obs_decoder'])
        if self.args.use_disc_model and (checkpoint['discount_model'] is not None):
            self.discount_model.load_state_dict(checkpoint['discount_model'])
        
        self.world_model_opt.load_state_dict(checkpoint['world_model_optimizer'])
        self.motor_actor_opt.load_state_dict(checkpoint['actor_optimizer'])
        self.motor_critic_opt.load_state_dict(checkpoint['critic_optimizer'])
        if self.args.sensor_mode == 'train':
            self.sensor_critic.load_state_dict(checkpoint['sensor_critic'])
            self.sensor_critic_target.load_state_dict(checkpoint['sensor_critic_target'])
            self.sensor_log_alpha = checkpoint['sensor_log_alpha']
            self.sensor_log_alpha_opt.load_state_dict(checkpoint['sensor_log_alpha_optimizer'])
            self.sensor_actor_opt.load_state_dict(checkpoint['sensor_actor_optimizer'])
            self.sensor_critic_opt.load_state_dict(checkpoint['sensor_critic_optimizer'])

    def set_step(self, step):
        self.step = step

    @torch.no_grad()
    def video_pred(self, obs, acs, nonterms, sensor_acs):
        '''
        Log images reconstructions
        '''
        T = obs.shape[0]
        obs = preprocess_obs(obs)
        obs_embed = self.obs_encoder(obs[1:]) # (T-1, n, e)
        
        init_state = self.rssm.init_state(4, self.device)
        _, states = self.rssm.observe_rollout(obs_embed[:5, :4], acs[:5, :4], nonterms[:5, :4], init_state, 5) # (5, 4, ...)
        recon = self.obs_decoder(torch.cat([self.rssm.get_feat(states), sensor_acs[:5, :4]], dim=-1)).mean # (5, 4, 3, 64, 64)

        init = {k: v[-1, :] for k, v in states.items()} # get the last posterior and imagine
        prior = self.rssm.imagine_rollout(acs[5:, :4], nonterms[5:, :4], init, T-5) # (45, 4, ...)
        features = self.rssm.get_feat(prior)
        openl = self.obs_decoder(torch.cat([features, sensor_acs[5:, :4]], dim=-1)).mean # (45, 4, 3, 64, 64)
        
        # select 6 envs, do 5 frames from data, rest reconstruct from dataset
        # so if dataset has 50 frames, 5 initial are real, 50-5 are imagined

        recon = recon.cpu()
        openl = openl.cpu()
        truth = obs[:, :4].cpu() + 0.5 # (50, 4, 3, 64, 64)

        if len(recon.shape)==3: #flat
            recon = recon.reshape(*recon.shape[:-1],*self.shape)
            openl = openl.reshape(*openl.shape[:-1],*self.shape)
            truth = truth.reshape(*truth.shape[:-1],*self.shape)


        model = torch.cat([recon[:5, :] + 0.5, openl + 0.5], 0)  # time
        error = (model - truth + 1) / 2
        video = torch.cat([truth, model, error], 3)  # on H
        T, B, C, H, W = video.shape  # time, batch, height, width, channels
        return video.permute(1, 0, 2, 3, 4)# reshape(T, C, H, B * W).permute(0, 2, 3, 1).numpy()
    
    def save_data(self, env, collect_steps, save_path):
        rews = self.act_and_collect_data(env, collect_steps)
        print(rews)
        np.savez(save_path, 
                 image=self.data_buffer.observations[self.data_buffer.idx-collect_steps:self.data_buffer.idx],
                 action=self.data_buffer.actions[self.data_buffer.idx-collect_steps:self.data_buffer.idx],
                 reward=self.data_buffer.rewards[self.data_buffer.idx-collect_steps:self.data_buffer.idx],
                 done=self.data_buffer.terminals[self.data_buffer.idx-collect_steps:self.data_buffer.idx])
        print("Save data to {}".format(save_path))