"""
Generating data from the CarRacing gym environment.
!!! DOES NOT WORK ON TITANIC, DO IT AT HOME, THEN SCP !!!
"""
import argparse
from os.path import join, exists
import gym
import numpy as np
from utils.misc import sample_continuous_policy
from cv2 import resize as rsz

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

def generate_data(rollouts, data_dir, noise_type): # pylint: disable=R0914
    """ Generates data """
    #assert exists(data_dir), "The data directory does not exist..."
    
    env = gym.make("CarRacing-v0")
    env = Env(env) 
    seq_len = 1000

    for i in range(rollouts):
        env.reset()
        env.env.viewer.window.dispatch_events()
        if noise_type == 'white':
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == 'brown':
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

        s_rollout = []
        r_rollout = []
        d_rollout = []

        t = 0
        while True:
            action = a_rollout[t]
            t += 1

            s, r, done, _ = env.step(action)
            env.env.viewer.window.dispatch_events()
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy)
