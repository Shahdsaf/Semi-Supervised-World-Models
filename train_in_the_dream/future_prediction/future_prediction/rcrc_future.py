

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
    filename = os.path.join(logdir, "run.log")
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, logdir


logger = logging.getLogger()
logdir = set_logger(logger)

logger.info("Running RCRC Reward Prediction")


import argparse

import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler, WeightedRandomSampler
#from environment import make_single_env
from torch.utils.tensorboard import SummaryWriter
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE, transform
from models import VAE
from cv2 import resize as rsz
import cv2

parser = argparse.ArgumentParser(description='Train a RCRC to predict rewards for the CarRacing-v0')
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

transition = np.dtype([('xesn_outs', np.float64, (1540,)), ('true', np.float64, (1,)),])  #+LSIZE




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
        #self.conv = Conv()
        vae_file = join("./vae", 'best.tar') 

        assert exists(vae_file) ,  "vae is untrained."

        vae_state = torch.load(vae_file, map_location={'cuda:0': str(device)})

        logger.info("Loading VAE at epoch {} with test loss {}".format(vae_state['epoch'], vae_state['precision']))

        self.vae = VAE(3, LSIZE).to(device).double()
        self.vae.load_state_dict(vae_state['state_dict'])
        
        self.W_in = nn.Linear(2*2*256+3, 512, bias=False)
        self.W = nn.Linear(512, 512, bias=False)
        self.W.weight.data = init_W(512, 512)
        self.x_esn = None
        self.alpha = alpha

    def forward(self, obs, prev_action):
        B = obs.shape[0]
        _, _, _, x_conv, _ = self.vae(obs)
        x_conv_flat = x_conv.view(B, -1)
        x_esn_input = torch.cat((x_conv_flat, prev_action), dim=1)
        
        if self.x_esn is None or self.x_esn.shape[0] != B:
            x_esn = torch.tanh(self.W_in(x_esn_input))
        else:
            x_hat = torch.tanh(self.W_in(x_esn_input) + self.W(self.x_esn))
            x_esn = (1 - self.alpha) * self.x_esn + self.alpha * x_hat
        self.x_esn = x_esn
        return (x_esn_input, x_esn)


class WM(nn.Module):
    def __init__(self, model):
        super(WM, self).__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.future = nn.Sequential(nn.Linear(1540, (1))) # +LSIZE
        
        
    def forward(self, inputs, action = None, z_out = False, xesn_out=False, future_out=False):
        
        if future_out:
            return self.future(inputs)
        
        if z_out:
            inputs = transform(inputs).unsqueeze(0).to(device).double()
            B = inputs.shape[0]
            _, _, _, _, zs = self.model.vae(inputs)
            zs = zs.view(B, -1)
            return zs
        
        if xesn_out:
            inputs = transform(inputs).unsqueeze(0).to(device).double()
            action = torch.from_numpy(action).unsqueeze(0).double().to(device)
            x_esn_input, x_esn = self.model(inputs, action)
            B = inputs.shape[0]
            S = torch.cat((x_esn_input, x_esn, torch.ones((B, 1)).double().to(device) ), dim=1)
            return S
    


class Agent():
    """
    Agent for training
    """
    buffer_capacity, batch_size = 2000, 500

    def __init__(self):
        self.training_step = 0
        self.fixed_model = FixedRandomModel(0.5).double().to(device)
        self.wm_model = WM(self.fixed_model).double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        self.loss = f.binary_cross_entropy_with_logits  #mse_loss # torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.wm_model.parameters(), lr=5e-3)
        self.loss_l = [1000000]
        self.pairs = [(0,0)]

    def save_param(self):
        torch.save(self.wm_model.state_dict(), 'param/ppo_net_params.pkl')

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

        true = torch.tensor(self.buffer['true'], dtype=torch.double).to(device).view(-1, 1)  #+LSIZE
        xesn_outs = torch.tensor(self.buffer['xesn_outs'], dtype=torch.double).to(device).view(-1, 1540)
        self.loss_l = []
        rew_pred = []
        rew_true = []
        #z_pred = []
        #z_true = []
        
        
        class_sample_count = np.unique(self.buffer['true'].squeeze(1), return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in self.buffer['true'].squeeze(1).astype(np.int)])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        #sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        #for index in BatchSampler(SequentialSampler(range(self.buffer_capacity)), self.batch_size, False):
        for index in BatchSampler(WeightedRandomSampler(samples_weight, self.buffer_capacity), self.batch_size, False):
            indices = []
            for idx, v in enumerate(self.buffer['true'].squeeze(1)[index]):
                if v ==1:
                    indices.append(idx)
            preds = agent.wm_model(xesn_outs[index], future_out=True)
            loss = self.loss(preds, true[index])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_l.append(loss.cpu().detach().numpy())
            if len(indices) > 0:
                rew_pred.append(preds.cpu().detach().numpy()[indices]) # (preds[0, -1])
                rew_true.append(self.buffer['true'].squeeze(1)[index][indices]) #(true[0, -1])
            #z_pred.append(preds[0, :-1])
            #z_true.append(true[0, :-1])
        
        #return torch.stack(z_pred), torch.stack(z_true), torch.tensor(rew_pred), torch.tensor(rew_true)
        #return torch.stack(rew_pred), torch.stack(rew_true)
        return rew_pred, rew_true

if __name__ == "__main__":
    agent = Agent()
    env = gym.make("CarRacing-v0")
    env = Env(env)
    parameters = sum(p.numel() for p in agent.wm_model.parameters())
    train_parameters = sum(p.numel() for p in agent.wm_model.parameters() if p.requires_grad)
    logger.info("Total Parameters : %s " % parameters)
    logger.info("Trainable Params : %s" % train_parameters)
    logger.info(agent.wm_model)
    
    if args.tb:
        writer = SummaryWriter(log_dir="./tb/")
    state = env.reset()
    
    min_loss = 1000000
    loss = [min_loss]
    for i_ep in range(25000):
        score = 0
        state = env.reset()
        agent.wm_model.model.x_esn = None
        
        for t in range(1000):
            action = env.action_space.sample()
            xesn_outs = agent.wm_model(state, action=action, xesn_out=True)
            state, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            #trues = agent.wm_model(state, z_out=True)
            
            #true_features = torch.cat((trues, torch.from_numpy(np.array([reward])).unsqueeze(0).double().to(device)), 
            #                          dim=1)
            
            #if reward <= 0 :
            #    reward = 0
            #else:
            #    reward = 1
            
            if done:
                done = 1
            else:
                done = 0
            
            true_features = torch.from_numpy(np.array([done])).unsqueeze(0).double().to(device)  
            
            
            if args.render:
                env.render()
            if agent.store((xesn_outs.cpu().numpy(), true_features.cpu().numpy())):
                logger.info('updating')
                rew_pred, rew_true = agent.update()
                
                '''
                z_pred, z_true, rew_pred, rew_true = agent.update()
                test_img = agent.wm_model.model.vae.decoder(z_pred)[0].cpu().detach().numpy()
                test_img = np.clip(test_img, 0, 1) * 255
                test_img = np.transpose(test_img, (1, 2, 0))
                #test_img = test_img.squeeze()
                test_img = test_img.astype(np.uint8)
                cv2.imwrite("./test_img_%s.png"%i_ep, test_img[:,:,0])

                test_img = agent.wm_model.model.vae.decoder(z_true)[0].cpu().detach().numpy()
                test_img = np.clip(test_img, 0, 1) * 255
                test_img = np.transpose(test_img, (1, 2, 0))
                #test_img = test_img.squeeze()
                test_img = test_img.astype(np.uint8)
                cv2.imwrite("./test_img_%s_true.png"%i_ep, test_img[:,:,0])
                '''
                logger.info("Predicted rewards : %s" % rew_pred)
                logger.info("True rewards : %s" % rew_true)
                
                loss = agent.loss_l
            
            if done:
                break
        
        
        if i_ep % args.log_interval == 0:
            if args.tb:
                writer.add_scalar(
                    "loss_avg", np.mean(loss), global_step=i_ep
                )
                
                
            logger.info('Ep {}\tLoss AVG: {:.2f}\t'.format(i_ep, np.mean(loss)))
            if np.mean(loss) < min_loss:
                min_loss = np.mean(loss)
                agent.save_param()
                logger.info("Saving a new model, min avg loss is {}".format(min_loss))
            
        #if np.mean(loss) < 1e-7:
        #    logger.info("Solved!")
        #    break

