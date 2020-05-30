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
    logdir = os.path.join("runs", current_time)

    os.makedirs(logdir, exist_ok=True)
    filename = os.path.join(logdir, "test.log")
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, logdir


logger = logging.getLogger()
logdir = set_logger(logger)

logger.info("Testing VM PPo Model")


import argparse

import numpy as np

import gym
import torch
import torch.nn as nn
from cv2 import resize as rsz
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE, transform
from models import MDRNNCell, VAE
from cv2 import resize as rsz
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=3, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=np.random.randint(np.int32(2**31-1)), metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


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
               


    def select_action(self, state, hidden):
        
        with torch.no_grad():
            _, latent_mu, _ = self.vae(state)
            alpha, beta = self.net(latent_mu, hidden[0])[0]
        
        action = alpha / (alpha + beta)

        
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        action = action.squeeze().cpu().numpy()
        return action, next_hidden
    
    def load_param(self):
        self.net.load_state_dict(torch.load('param/ppo_net_params.pkl'))

    


if __name__ == "__main__":
    agent = Agent()
    agent.load_param()
    env = gym.make("CarRacing-v0")
    env = Env(env)

    running_score = []
    state = env.reset()
    
    for i_ep in range(100):
        score = 0
        state = env.reset()
        hidden = [
            torch.zeros(1, RSIZE).to(device).double()
            for _ in range(2)]
        for t in range(1000):
            state = transform(state).unsqueeze(0).to(device).double()
            action, hidden = agent.select_action(state, hidden)
            state, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            if done:
                break
        
        running_score.append(score) 

        logger.info('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
    logger.info('Avg Score: {} + {}'.format(np.mean(running_score), np.std(running_score)))
