# Ideas

## Multi-agent Reinforcement Learning

Instead of having a single agent, in the multi-agent setting, we have multiple agents interacting with each other and the environment. They can have a single common goal (Cooperative environment) or compete on the limited resources. You can expand this formulation to many applications (from natural language processing to computer networks or game theory).

A simple example of these applications is [pistonball](https://www.pettingzoo.ml/butterfly/pistonball) game in which we have some pistons that try to move a ball from right to left. You can make the problem more difficult by adding obstacles or new actions to the agents (pistons).

The more interesting multi-agent games can be found [here](https://www.pettingzoo.ml/).

## Transfer learning in Reinforcement Learning

It is quite helpful to employ external knowledge earned in other environments and by other agents in our problem. To this aim, we need to initialize our neural networks with the pre-trained models. For instance, an agent's experience in the PuckWorld game can be used by another one in the Snake game. You can find more details about this example in this [paper](https://web.stanford.edu/class/cs234/past_projects/2017/2017_Asawa_Elamri_Pan_Transfer_Learning_Paper.pdf) (they also included their source code).

There are also many environments in [minigrid](https://github.com/maximecb/gym-minigrid) which are suitable for transfer learning.

## Deep Reinforcement Learning for Traffic Signal Control

Deep RL can be applied to the complex traffic congestion problem in order to decrease travel time, queue length, or the number of stops. The congestion control problem is so complex that we can handle it in 3 weeks. However, some simulators make working on this problem easier for us. You can find them and more materials on this subject, including tutorials, slides, papers, and simulators, [here](https://traffic-signal-control.github.io/#tutorial).

## Other resources

[RL baselines zoo](https://opensourcelibs.com/lib/rl-baselines-zoo)  
[OpenAI Gym](https://opensourcelibs.com/libs/openai-gym)  


