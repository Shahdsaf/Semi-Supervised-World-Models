import numpy as np
import gym

from cv2 import resize as rsz
from gym.spaces.box import Box

SCREEN_X = 64
SCREEN_Y = 64

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

# from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
if __name__=="__main__":
  from pyglet.window import key
  a = np.array( [0.0, 0.0, 0.0] )
  def key_press(k, mod):
    global restart
    if k==0xff0d: restart = True
    if k==key.LEFT:  a[0] = -1.0
    if k==key.RIGHT: a[0] = +1.0
    if k==key.UP:    a[1] = +1.0
    if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
  def key_release(k, mod):
    if k==key.LEFT  and a[0]==-1.0: a[0] = 0
    if k==key.RIGHT and a[0]==+1.0: a[0] = 0
    if k==key.UP:    a[1] = 0
    if k==key.DOWN:  a[2] = 0
  env = gym.make("CarRacing-v0")
  env = Env(env) 
  env.render()
  env.viewer.window.on_key_press = key_press
  env.viewer.window.on_key_release = key_release
  while True:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
      s, r, done, info = env.step(a)
      total_reward += r
      if steps % 200 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
      steps += 1
      env.render()
      if done or restart: break
  env.monitor.close()