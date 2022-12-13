import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import matplotlib.pyplot as plt

from anim import Anim, quat


""" load animation data """

anim = Anim.load_bvh('./data/walk1_subject5.bvh')

""" preprocessing """

anim.add_root()
# anim.delete_root_motion()

anim_arr = anim[200:].split(32)
anim_arr.reset_origin_to_first_frame()

# from viewer import Viewer
# anim = anim_arr.join()
# anim.fk()
# v = Viewer(1200, 800)
# v.add_anim(np.asarray([0, 0, 0]), anim)
# v.show()

# exit()

rotations = quat.to_scaled_angle_axis(anim_arr.rotations)

nsamples, nframes, nbones, channels = rotations.shape

rotations = (rotations + np.pi) / np.pi / 2

plt.figure()
plt.imshow(rotations[0], interpolation='none')
plt.show()

print(rotations[0])

rotations = np.pad(rotations, [(0, 0), (0, 32 - nframes), (0, 32 - nbones), (0, 0)])
rotations = np.moveaxis(rotations, -1, 1)

""" training """

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# https://github.com/lucidrains/denoising-diffusion-pytorch/
# https://blog.csdn.net/Eric_1993/article/details/127455977

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Unet(
    dim = 32,
    dim_mults = (1, 4, 8),
    channels = 3
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 10000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).to(device)

training_anims = torch.tensor(rotations[:160], dtype=torch.float16, device=device)
loss = diffusion(training_anims)
loss.backward()

sampled_anims = diffusion.sample(batch_size = 4).cpu().numpy()

""" post-processing """

# sampled_anims = rotations[10:14]

sampled_anims = sampled_anims[..., :nframes, :nbones]
sampled_anims = np.moveaxis(sampled_anims, 1, -1)

plt.figure()
plt.imshow(sampled_anims[0], interpolation='none')
plt.show()

""" generate anim objects """

anims = []
for i in range(len(sampled_anims)):
    a = anim.copy()
    a.name = 'sample_' + str(i)
    a.positions = a.positions[:nframes]
    a.rotations = quat.from_scaled_angle_axis(sampled_anims[i] * 2 * np.pi - np.pi)
    a.fk()
    anims.append(a)

""" viewer """

from anim.viewer import Viewer

v = Viewer(1200, 800)

for i, anim in enumerate(anims):
    v.add_anim(np.asarray([i, 0, 0]), anim)

v.show()
