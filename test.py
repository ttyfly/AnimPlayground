from pyray import *
import numpy as np
from anim import Anim

# https://electronstudio.github.io/raylib-python-cffi/

from viewer import Viewer

anim_1 = Anim.load_bvh('data/dataset-1_walk_normal_001.bvh')
anim_1.name = 'walk'
anim_1.delete_root()
anim_1.add_root()
anim_1.delete_root_motion()

anim_2 = Anim.load_bvh('data/dataset-1_punch_normal_001.bvh')
anim_2.name = 'punch'
anim_2.delete_root()
anim_2.add_root()
anim_2.delete_root_motion()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

upper_body = ['Spine', 'Chest', 'Neck', 'Head', 'Shoulder_L', 'UpperArm_L',
              'LowerArm_L', 'Hand_L', 'Shoulder_R', 'UpperArm_R', 'LowerArm_R', 'Hand_R']

upper_body_musk = np.asarray([1 if name in upper_body else 0 for name in anim_1.bone_names])

x = np.arange(16)
blend_curve = sigmoid(0.3 * (x - 8))
blend_curve = (blend_curve - np.min(blend_curve)) / (np.max(blend_curve) - np.min(blend_curve))
# blend_alpha = blend_curve[..., np.newaxis] * upper_body_musk

# anim = Anim.blend(anim_1, anim_2, blend_alpha)

anim = anim_1[:56] + Anim.blend(anim_1[56:72], anim_2[0:16], blend_curve) + anim_2[20:]

v = Viewer(1200, 800)

# v.add_anim(np.asarray([-2, 0, 0]), anim_1)
# v.add_anim(np.asarray([-1, 0, 0]), anim_2)
v.add_anim(np.asarray([0, 0, 0]), anim)

v.add_curve(blend_curve, 0, 1, start_frame=56)

v.add_frame_tag(56)
v.add_frame_tag(72)

# def interpolate(anim: Anim, keys: List[int]):
#     for i in range(len(keys) - 1):
#         anim.positions[keys[i]:keys[i + 1] + 1] = np.linspace(anim.positions[keys[i]], anim.positions[keys[i + 1]], keys[i + 1] - keys[i] + 1)
#         anim.rotations[keys[i]:keys[i + 1] + 1] = quat.sinterpolate(anim.rotations[keys[i]], anim.rotations[keys[i + 1]], keys[i + 1] - keys[i] + 1)

v.show()
