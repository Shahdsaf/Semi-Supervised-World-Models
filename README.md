# World Models with PyTorch
A new version of world models using Echo-state networks and random weight-fixed CNNs in Pytorch. Also, the controller leverages RL algorithms, e.g. PPO methods.

## Requirement
To run the code, you need
- [pytorch](https://pytorch.org/)
- [gym](https://github.com/openai/gym)

## Method
Every action will be repeated for 8 frames. To get velocity information, state is defined as adjacent 4 frames in shape (4, 96, 96). Use a two heads FCN to represent the actor and critic respectively. The actor outputs α, β for each actin as the parameters of Beta distribution. 
<div align=center><img src="img/network.png" width="30%" /></div>

## Training
Start a Visdom server with ```python -m visdom.server```, it will serve http://localhost:8097/ by default.

To train the agent, run```python train.py --render --vis``` or ```python train.py --render``` without visdom. 
To test, run ```python test.py --render```.

## Performance
<div align=center><img src="img/car_racing_ppo.png" width="50%"/></div>
<div align=center><img src="img/car_racing_demo_ppo.gif"/></div>

