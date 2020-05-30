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
    filename = os.path.join(logdir, "run.log")
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, logdir


logger = logging.getLogger()
logdir = set_logger(logger)

logger.info("Running RCRC PPO")


import argparse

import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
#from environment import make_single_env
from torch.utils.tensorboard import SummaryWriter
from cv2 import resize as rsz


parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=3, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=np.random.randint(np.int32(2**31-1)), metavar='N', help='random seed (default: 0)')
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

transition = np.dtype([('s', np.float64, (1025,)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (1025,))])




    

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
        S = torch.cat((x_conv, x_esn, torch.ones((B, 1)).double().to(device) ), dim=1)

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
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self):
        self.training_step = 0
        self.fixed_model = FixedRandomModel(0.5).double().to(device)
        self.wm_model = WM(self.fixed_model).double().to(device)
        self.net = ACC(self.wm_model).double().to(device)
        if os.path.exists("param/ppo_net_params.pkl"):
            self.load_param()
            logger.info("Model Loaded Successfully")
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            (alpha, beta), _, rcrc_s = self.net(state)
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp, rcrc_s

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
    def load_param(self):
        self.net.load_state_dict(torch.load('param/ppo_net_params.pkl'))

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)
        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_, actual_obs=False)[1]
            adv = target_v - self.net(s, actual_obs=False)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index], actual_obs=False)[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index], actual_obs=False)[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()


if __name__ == "__main__":
    agent = Agent()
    env = gym.make("CarRacing-v0")
    env = Env(env, resize = True)
    parameters = sum(p.numel() for p in agent.net.parameters())
    train_parameters = sum(p.numel() for p in agent.net.parameters() if p.requires_grad)
    logger.info("Total Parameters : %s " % parameters)
    logger.info("Trainable Params : %s" % train_parameters)
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
        agent.net.wm_model.model.x_esn = None
        action, a_logp, rcrc_s = agent.select_action(state)
        for t in range(1000):
            state, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            next_action, next_a_logp, rcrc_s_ = agent.select_action(state)
            if args.render:
                env.render()
            if agent.store((rcrc_s.cpu().numpy(), action, a_logp, reward, rcrc_s_.cpu().numpy())):
                logger.info('updating')
                agent.update()
            score += reward
            rcrc_s = rcrc_s_
            action = next_action
            a_logp = next_a_logp
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
