from typing import List
import re
import os.path

import numpy as np
from . import quat


class Skeleton(object):

    name: str
    parents: np.ndarray
    bone_names: List[str]

    def __init__(
        self,
        name: str,
        bone_names: List[str],
        parents: np.ndarray
    ):
        self.name = name
        self.bone_names = bone_names
        self.parents = parents

    def copy(self):
        return Skeleton(self.name, self.bone_names.copy(), self.parents.copy())

    def __eq__(self, other):
        if isinstance(other, Skeleton):
            return (
                self is other
                or (
                    np.asarray(self.parents == other.parents).all()
                    and self.bone_names == other.bone_names
                )
            )
        else:
            return False


class AnimArray(object):

    names: List[str]
    rotations: np.ndarray
    positions: np.ndarray
    skeleton: Skeleton
    frame_time: float

    global_rotations: np.ndarray
    global_positions: np.ndarray

    def __init__(
        self,
        skeleton: Skeleton,
        frame_time: float = 0.033333,
        initial_names = [],
        initial_rotations: np.ndarray = np.empty((0, 0, 0, 4)),
        initial_positions: np.ndarray = np.empty((0, 0, 0, 3))
    ):
        assert initial_rotations.ndim == 4 and initial_rotations.shape[-1] == 4
        assert initial_positions.ndim == 4 and initial_positions.shape[-1] == 3
        assert len(initial_positions) == len(initial_names)
        assert len(initial_rotations) == len(initial_names)

        self.skeleton = skeleton
        self.frame_time = frame_time
        self.names = initial_names
        self.rotations = initial_rotations
        self.positions = initial_positions

    def __len__(self):
        return len(self.names)

    def __getitem__(self, key):
        if isinstance(key, int):
            return Anim(
                self.names[key],
                self.skeleton,
                self.rotations[key],
                self.positions[key],
                self.frame_time
            )
        else:
            raise KeyError(key)

    def fk(self):
        self.global_rotations, self.global_positions = quat.fk(self.rotations, self.positions, self.skeleton.parents)

    def append(self, anim: 'Anim'):
        assert self.skeleton == anim.skeleton
        assert self.frame_time == anim.frame_time

        self.names.append(anim.name)

        self.rotations = np.append(self.rotations, anim.rotations[np.newaxis, ...], axis=0)
        self.positions = np.append(self.positions, anim.positions[np.newaxis, ...], axis=0)

    def reset_origin(self, rot: np.ndarray, pos: np.ndarray):
        assert rot.shape == (4,)
        assert pos.shape == (3,)

        self.positions[:, :, :1] -= quat.mul_vec(
            quat.inv(rot),
            self.positions[:, :, :1] - pos
        )
        self.rotations[:, :, :1] = quat.mul(quat.inv(rot), self.rotations[:, :, :1])

    def reset_origin_to_first_frame(self):
        self.positions[:, :, :1] = quat.mul_vec(
            quat.inv(self.rotations[:, :1, :1]),
            self.positions[:, :, :1] - self.positions[:, :1, :1]
        )
        self.rotations[:, :, :1] = quat.mul(quat.inv(self.rotations[:, :1, :1]), self.rotations[:, :, :1])

    def join(self) -> 'Anim':
        return Anim(
            self.names[0],
            self.skeleton,
            self.rotations.reshape((-1, self.rotations.shape[2], 4)),
            self.positions.reshape((-1, self.positions.shape[2], 3)),
            self.frame_time
        )


class Anim(object):

    name: str
    rotations: np.ndarray
    positions: np.ndarray
    skeleton: Skeleton
    frame_time: float

    global_rotations: np.ndarray
    global_positions: np.ndarray

    def __init__(
        self,
        name: str,
        skeleton: Skeleton,
        rotations: np.ndarray = np.empty((0, 0, 4)),
        positions: np.ndarray = np.empty((0, 0, 3)),
        frame_time: float = 0.033333  # default to be 30 fps
    ):
        assert len(rotations) == len(positions)

        self.name = name
        self.skeleton = skeleton
        self.rotations = rotations
        self.positions = positions
        self.frame_time = frame_time

    def delete_root_motion(self):
        self.positions[:, 0] = np.zeros_like(self.positions[:, 0])

    def delete_root(self):
        self.positions[:, 1] = quat.mul_vec(self.rotations[:, 0], self.positions[:, 1]) + self.positions[:, 0]
        self.rotations[:, 1] = quat.mul(self.rotations[:, 0], self.rotations[:, 1])

        self.skeleton = Skeleton(
            self.skeleton.name,
            self.skeleton.bone_names[1:],
            self.skeleton.parents[1:] - 1
        )

        self.positions = self.positions[:, 1:]
        self.rotations = self.rotations[:, 1:]

    def add_root(self):
        root_position = np.asarray([1, 0, 1]) * self.positions[:, :1]
        root_direction = np.asarray([1, 0, 1]) * quat.mul_vec(self.rotations[:, :1], np.asarray([0, 1, 0]))
        root_rotation = quat.normalize(quat.between(np.asarray([0, 0, 1]), root_direction))

        self.positions[:, :1] = quat.mul_vec(quat.inv(root_rotation), self.positions[:, :1] - root_position)
        self.rotations[:, :1] = quat.mul(quat.inv(root_rotation), self.rotations[:, :1])

        self.positions = np.concatenate([root_position, self.positions], axis=1)
        self.rotations = quat.unroll(np.concatenate([root_rotation, self.rotations], axis=1))

        self.skeleton = Skeleton(
            self.skeleton.name,
            ['Root'] + self.skeleton.bone_names,
            np.concatenate([[-1], self.skeleton.parents + 1])
        )

    def split(self, clip_length: int, truncation: bool = True) -> AnimArray:
        rotations: np.ndarray
        positions: np.ndarray

        if truncation:
            rotations = self.rotations[:len(self.rotations) - len(self.rotations) % clip_length]
            positions = self.positions[:len(self.positions) - len(self.positions) % clip_length]
        else:
            rotations = np.pad(self.rotations, (0, clip_length - len(self.rotations) % clip_length), 'edge')
            positions = np.pad(self.positions, (0, clip_length - len(self.positions) % clip_length), 'edge')

        clip_count = len(self.rotations) // clip_length

        return AnimArray(
            self.skeleton,
            self.frame_time,
            initial_names=[str.format('{}_{}', self.name, i + 1) for i in range(clip_count)],
            initial_rotations=rotations.reshape((clip_count, clip_length, -1, 4)),
            initial_positions=positions.reshape((clip_count, clip_length, -1, 3))
        )

    def reset_origin(self, rot: np.ndarray, pos: np.ndarray):
        assert rot.shape == (4,)
        assert pos.shape == (3,)

        self.positions[:, :1] -= quat.mul_vec(
            quat.inv(rot),
            self.positions[:, :1] - pos
        )
        self.rotations[:, :1] = quat.mul(quat.inv(rot), self.rotations[:, :1])

    def reset_origin_to_first_frame(self):
        self.reset_origin(self.rotations[0, 0], self.positions[0, 0])

    def fk(self):
        self.global_rotations, self.global_positions = quat.fk(self.rotations, self.positions, self.skeleton.parents)

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, key):
        if isinstance(key, slice):
            anim = self.copy()
            anim.positions = anim.positions[key]
            anim.rotations = anim.rotations[key]
            return anim
        else:
            raise KeyError(key)

    def __add__(self, obj):
        if isinstance(obj, Anim):
            assert self.skeleton == obj.skeleton
            assert self.frame_time == obj.frame_time

            return Anim(
                str.format('{}_{}', self.name, obj.name),
                self.skeleton,
                np.concatenate([self.rotations, obj.rotations]),
                np.concatenate([self.positions, obj.positions]),
                self.frame_time
            )
        else:
            raise TypeError(obj)

    def copy(self):
        return Anim(
            self.name,
            self.skeleton,
            self.rotations.copy(),
            self.positions.copy(),
            self.frame_time
        )

    @staticmethod
    def load_bvh(filename: str, rot_order = None) -> 'Anim':
        channelmap = {
            'Xrotation': 'x',
            'Yrotation': 'y',
            'Zrotation': 'z'
        }

        channelmap_inv = {
            'x': 'Xrotation',
            'y': 'Yrotation',
            'z': 'Zrotation',
        }

        f = open(filename, "r")

        i = 0
        active = -1
        end_site = False

        bone_names = []
        orients = np.empty((0, 4))
        offsets = np.empty((0, 3))
        parents = np.empty((0,), dtype=int)
        frame_time = 0.033333

        channels = 3
        positions = np.empty((0, 0, 3))
        rotations = np.empty((0, 0, 4))

        # Parse the  file, line by line
        for line in f:

            if "HIERARCHY" in line: continue
            if "MOTION" in line: continue

            rmatch = re.match(r"ROOT ([\w:]+)", line)
            if rmatch:
                bone_names.append(rmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if "{" in line: continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    active = parents[active]
                continue

            offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if offmatch:
                if not end_site:
                    offsets[active] = np.array([list(map(float, offmatch.groups()))])
                continue

            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                if rot_order is None:
                    channelis = 0 if channels == 3 else 3
                    channelie = 3 if channels == 3 else 6
                    parts = line.split()[2 + channelis:2 + channelie]
                    if any([p not in channelmap for p in parts]):
                        continue
                    rot_order = "".join([channelmap[p] for p in parts])
                continue

            jmatch = re.match(r"\s*JOINT\s+([\w:]+)", line)
            if jmatch:
                bone_names.append(jmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if "End Site" in line:
                end_site = True
                continue

            fmatch = re.match(r"\s*Frames:\s+(\d+)", line)
            if fmatch:
                fnum = int(fmatch.group(1))
                positions = offsets[np.newaxis].repeat(fnum, axis=0)
                rotations = np.zeros((fnum, len(orients), 3))
                continue

            fmatch = re.match(r"\s*Frame Time:\s+([\d\.]+)", line)
            if fmatch:
                frame_time = float(fmatch.group(1))
                continue

            dmatch = line.strip().split(' ')
            if dmatch:
                data_block = np.array(list(map(float, dmatch)))
                N = len(parents)
                fi = i
                if channels == 3:
                    positions[fi, 0:1] = data_block[0:3]  # root motion
                    rotations[fi, :] = data_block[3:].reshape(N, 3)
                elif channels == 6:
                    data_block = data_block.reshape(N, 6)
                    positions[fi, :] = data_block[:, 0:3]
                    rotations[fi, :] = data_block[:, 3:6]
                elif channels == 9:
                    positions[fi, 0] = data_block[0:3]
                    data_block = data_block[3:].reshape(N - 1, 9)
                    rotations[fi, 1:] = data_block[:, 3:6]
                    positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]  # position * scale
                else:
                    raise Exception("Too many channels! %i" % channels)

                i += 1

        f.close()

        assert rot_order is not None

        anim_name = os.path.basename(filename)

        sk = Skeleton(
            str.format('{}_sk', anim_name),
            bone_names,
            parents
        )

        return Anim(
            anim_name,
            sk,
            quat.unroll(quat.from_euler(np.radians(rotations), order=rot_order)),
            positions * 0.01,
            frame_time
        )

    @staticmethod
    def blend(anim_1: 'Anim', anim_2: 'Anim', alpha):
        alpha = np.asarray(alpha)

        assert alpha.ndim in [0, 1, 2]
        assert anim_1.skeleton == anim_2.skeleton
        assert anim_1.frame_time == anim_2.frame_time

        return Anim(
            str.format('{}_{}_blend', anim_1.name, anim_2.name),
            anim_1.skeleton,
            quat.slerp(anim_1.rotations, anim_2.rotations, alpha),
            quat.vec_lerp(anim_1.positions, anim_2.positions, alpha),
            anim_1.frame_time
        )
