import logging
import sys
import datetime
import os 
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

logger.info("Testing RCRC PPO")


import argparse

import numpy as np

import gym
import torch
import torch.nn as nn
from cv2 import resize as rsz

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
        #out_img_stack = (out_img_stack / out_img_stack.max()) * 255 
        #out_img_stack = out_img_stack.astype(np.uint8).transpose(1,2,0)
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
        #out_img_stack = (out_img_stack / out_img_stack.max()) * 255 
        #out_img_stack = out_img_stack.astype(np.uint8).transpose(1,2,0)
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

    
class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 32, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 8, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 2, stride=2)

    def forward(self, image_stack):
        x = self.conv1(image_stack)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    




def init_W(n, m):
    weight = torch.normal(mean=torch.zeros((n, m)), std=torch.ones((n, m)))

    N = n * m
    p = int(0.2 * N)

    u, s, v = torch.svd(weight, compute_uv=True)
    s_ = 0.95 * s / s.max()

    weight = u * s_ * v.t()
    indices = np.random.choice(N, p)
    for i in indices:
        a = i // n
        b = i - a * n
        weight[a, b] = 0
    return weight


class FixedRandomModel(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.conv = Conv()
        self.W_in = nn.Linear(512, 512, bias=False)
        self.W = nn.Linear(512, 512, bias=False)
        self.W.weight.data = init_W(512, 512)
        self.x_esn = None
        self.alpha = alpha

    def forward(self, obs):
        B = obs.shape[0]
        x_conv = self.conv(obs)
        x_conv_flat = x_conv.view(B, -1)

        if self.x_esn is None or self.x_esn.shape[0] != B:
            x_esn = torch.tanh(self.W_in(x_conv_flat))
        else:
            x_hat = torch.tanh(self.W_in(x_conv_flat) + self.W(self.x_esn))
            x_esn = (1 - self.alpha) * self.x_esn + self.alpha * x_hat
        self.x_esn = x_esn
        return (x_conv_flat, x_esn)


class WM(nn.Module):
    def __init__(self, model):
        super(WM, self).__init__()
        self.model = model
        
        
    def forward(self, obs):
        x_conv, x_esn = self.model(obs)
        B = obs.shape[0]
        S = torch.cat((x_conv, x_esn, torch.ones((B, 1)).float().to(device) ), dim=1)

        return S
    
    
class ACC(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, wm_model):
        super(ACC, self).__init__()
        self.wm_model = wm_model
        for p in self.wm_model.parameters():
            p.requires_grad = False
        self.v = nn.Sequential(nn.Linear(1025, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(1025, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())


    def forward(self, x, actual_obs=True):
        if actual_obs:
            x = self.wm_model(x)
        rcrc_s = x.view(-1, 1025)
        v = self.v(rcrc_s)
        p = self.fc(rcrc_s)
        alpha = self.alpha_head(p) + 1
        beta = self.beta_head(p) + 1

        return (alpha, beta), v, rcrc_s




class Agent():
    """
    Agent for testing
    """

    def __init__(self):
        self.fixed_model = FixedRandomModel(0.5).float().to(device)
        self.wm_model = WM(self.fixed_model).float().to(device)
        self.net = ACC(self.wm_model).float().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action
    
    
    def load_param(self):
        self.net.load_state_dict(torch.load('param/ppo_net_params.pkl'))


if __name__ == "__main__":
    agent = Agent()
    agent.load_param()
    env = gym.make("CarRacing-v0")
    env = Env(env, resize = True)

    running_score = []
    state = env.reset()
    for i_ep in range(100):
        score = 0
        state = env.reset()
        agent.net.wm_model.model.x_esn = None
        for t in range(1000):
            action = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            state = state_
            if done:
                break
        
        running_score.append(score) 

        logger.info('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
    logger.info('Avg Score: {} + {}'.format(np.mean(running_score), np.std(running_score)))
