import gym
import os
import random
from PIL import Image

env = gym.make('LunarLander-v2')
PATH = 'sequences/1'
try:
	os.makedirs(PATH)
except:
	pass
for j in range(500):
    env.reset()
    LPATH = PATH + '/seq_' + str(j)
    os.makedirs(LPATH)
    actions = []
    for i in range(150):
        img = Image.fromarray(env.render(mode='rgb_array'), 'RGB')
        img.save(LPATH + '/' + str(i) + '.png')
        k = 2
        if (random.randint(0,2) != 0) or (i < 10):
            while k == 2:
                k = env.action_space.sample()
        actions.append(k)
        observation = env.step(k)
    with open(LPATH + '/actions.txt', 'w') as f:
        for action in actions:
            f.write(str(action))

