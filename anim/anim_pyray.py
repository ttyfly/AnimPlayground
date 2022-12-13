from pyray import *
import numpy as np

from .anim import Anim


def draw_anim(root: Vector3, anim: Anim, frame: int):
    gpos = anim.global_positions + np.asarray([root.x, root.y, root.z])
    parents = anim.skeleton.parents

    nframes = gpos.shape[0]
    nbones = gpos.shape[1]

    frame = int(clamp(frame, 0, nframes - 1))

    for j in range(nbones):
        pos = Vector3(gpos[frame, j, 0], gpos[frame, j, 1], gpos[frame, j, 2])
        draw_sphere(pos, 0.01, BLACK)

    for j in range(nbones):
        if parents[j] < 0:
            continue
        pos = Vector3(gpos[frame, j, 0], gpos[frame, j, 1], gpos[frame, j, 2])
        parent_pos = Vector3(gpos[frame, parents[j], 0], gpos[frame, parents[j], 1], gpos[frame, parents[j], 2])
        draw_line_3d(parent_pos, pos, BLACK)
