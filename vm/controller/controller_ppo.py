import logging
import sys
import datetime
import os 
from os.path import join, exists

def set_logger(logger):
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s"
    )

    logger.setLevel(logging.INFO)
    logger.handlers = []

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.join("logs", current_time)

    os.makedirs(logdir, exist_ok=True)
    filename = os.path.join(logdir, "cntlr.log")
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, logdir


logger = logging.getLogger()
logdir = set_logger(logger)

logger.info("Training Controller PPO")



import argparse

import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE, transform
from models import MDRNNCell, VAE
from cv2 import resize as rsz
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=3, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--tb', action='store_true', help='use tb')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

transition = np.dtype([('mu', np.float64, (LSIZE)), ('hidden', np.float64, (1, RSIZE)), ('a', np.float64, (3,)), ('a_logp', np.float64), ('r', np.float64), ('mu_', np.float64, (LSIZE)), ('hidden_', np.float64, (1, RSIZE))])



class Env(gym.Wrapper):
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self, env, resize=False, img_stack=3, action_repeat=8):
        super(Env, self).__init__(env)
        self.env = env
        #self.env.seed(args.seed)
        self.reward_threshold = self.env.spec.reward_threshold
        self.resize = resize
        self.img_stack = img_stack
        self.action_repeat = action_repeat

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        if self.resize:
            img_gray = rsz(img_gray, (64,64))
        self.stack = [img_gray] * self.img_stack  # four frames for decision
        out_img_stack = np.array(self.stack).astype(np.float64) 
        #out_img_stack = np.interp(out_img_stack, (out_img_stack.min(), out_img_stack.max()), (0, 255))
        out_img_stack = (out_img_stack / out_img_stack.max()) * 255 
        out_img_stack = out_img_stack.astype(np.uint8).transpose(1,2,0)
        return out_img_stack

    def step(self, action):
        
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        if self.resize:
            img_gray = rsz(img_gray, (64,64))
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        if done or die:
            done = True
        out_img_stack = np.array(self.stack).astype(np.float64) 
        #out_img_stack = np.interp(out_img_stack, (out_img_stack.min(), out_img_stack.max()), (0, 255))
        out_img_stack = (out_img_stack / out_img_stack.max()) * 255 
        out_img_stack = out_img_stack.astype(np.uint8).transpose(1,2,0)
        
        return out_img_stack, total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
    
    
class Controller(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, latents, recurrents, actions):
        super(Controller, self).__init__()
        self.input_size = latents + recurrents
        self.v = nn.Sequential(nn.Linear(self.input_size, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(self.input_size, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, actions), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, actions), nn.Softplus())


    def forward(self, *inputs):
        #x = self.cnn_base(x)
        #x = x.view(-1, self.input_size)
        cat_in = torch.cat(inputs, dim=1)
        v = self.v(cat_in)
        x = self.fc(cat_in)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

    
    
class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 1500, 128

    def __init__(self):
        
        
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file = \
            [join("./training", m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            logger.info("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device).double()
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device).double()
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
    
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.mdrnn.parameters():
            p.requires_grad = False
        
        
        self.net = Controller(LSIZE, RSIZE, ASIZE).to(device).double()
        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            logger.info("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.net.load_state_dict(ctrl_state['state_dict'])
        
        self.training_step = 0
       
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state, hidden):
        
        with torch.no_grad():
            _, latent_mu, _ = self.vae(state)
            alpha, beta = self.net(latent_mu, hidden[0])[0]
        
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        a_logp = a_logp.item()
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        
        return action.squeeze().cpu().numpy(), a_logp, latent_mu, next_hidden

    def save_param(self):
        torch.save(self.net.state_dict(), 'param/ppo_net_params.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1
        mu = torch.tensor(self.buffer['mu'], dtype=torch.double).to(device)
        hidden = torch.tensor(self.buffer['hidden'], dtype=torch.double).to(device).view(-1, RSIZE)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        mu_ = torch.tensor(self.buffer['mu_'], dtype=torch.double).to(device)
        hidden_ = torch.tensor(self.buffer['hidden_'], dtype=torch.double).to(device).view(-1, RSIZE)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + args.gamma * self.net(mu_, hidden_)[1]
            adv = target_v - self.net(mu, hidden)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(mu[index], hidden[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(mu[index], hidden[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()


if __name__ == "__main__":
    
    env = gym.make("CarRacing-v0")
    env = Env(env) 
    
    agent = Agent()
    
    
    #parameters = sum(p.numel() for p in agent.net.parameters())
    #train_parameters = sum(p.numel() for p in agent.net.parameters() if p.requires_grad)
    #logger.info("Total Parameters : %s " % parameters)
    #logger.info("Trainable Params : %s" % train_parameters)
    logger.info(agent.net)
    
    if args.tb:
        writer = SummaryWriter(log_dir="./tb/")
    training_records = []
    running_score = 0
    state = env.reset()
    max_score = -1e4
    for i_ep in range(100000):
        score = 0
        state = env.reset()
        state = transform(state).unsqueeze(0).to(device).double()
        hidden = [
            torch.zeros(1, RSIZE).to(device).double()
            for _ in range(2)]

        for t in range(1000):
            action, a_logp, latent_mu, hidden_ = agent.select_action(state, hidden)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            state = transform(state_).unsqueeze(0).to(device).double()
            with torch.no_grad():
                _, latent_mu_, _ = agent.vae(state) 
            if args.render:
                env.render()
            if agent.store((latent_mu[0].cpu(), hidden[0].cpu(), action, a_logp, reward, latent_mu_[0].cpu(), hidden_[0].cpu())):
                logger.info('updating')
                agent.update()
            score += reward
            hidden = hidden_
            if done:
                break
        running_score = running_score * 0.99 + score * 0.01
        
        if i_ep % args.log_interval == 0:
            if args.tb:
                writer.add_scalar(
                    "running_score", running_score, global_step=i_ep
                )
                writer.add_scalar(
                    "last_score", score, global_step=i_ep
                )
            logger.info('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            if running_score > max_score:
                max_score = running_score
                agent.save_param()
                logger.info("Saving a new model, max score is {}".format(max_score))
            
        if running_score > env.reward_threshold:
            logger.info("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break
