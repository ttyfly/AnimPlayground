import re
import os.path
from typing import Dict
import numpy as np
import quat


class Skeleton(object):
    def __init__(self) -> None:
        self.offsets = np.array([]).reshape([0, 3])
        self.parents = np.array([], dtype=int)
        self.bone_names = []


class Anim(object):
    def __init__(self, name: str) -> None:
        self.name = name

        self.positions = np.array([]).reshape((0, 0, 3))
        self.rotations = np.array([]).reshape((0, 0, 4))
        self.parents = np.array([], dtype=int)
        self.bone_names = []
        self.frame_time = 0.033333  # default to be 30 fps

        self.gdirty = True
        self._gpos = np.array([]).reshape((0, 0, 3))
        self._grot = np.array([]).reshape((0, 0, 4))

    def delete_root_motion(self):
        self.positions[:, 0] = np.zeros_like(self.positions[:, 0])

    def delete_root(self):
        self.positions[:, 1] = quat.mul_vec(self.rotations[:, 0], self.positions[:, 1]) + self.positions[:, 0]
        self.rotations[:, 1] = quat.mul(self.rotations[:, 0], self.rotations[:, 1])

        self.bone_names = self.bone_names[1:]
        self.parents = self.parents[1:] - 1
        self.positions = self.positions[:, 1:]
        self.rotations = self.rotations[:, 1:]

    def add_root(self):
        root_position = np.asarray([1, 0, 1]) * self.positions[:, :1]
        root_direction = np.asarray([1, 0, 1]) * quat.mul_vec(self.rotations[:, :1], np.asarray([0, 1, 0]))
        root_rotation = quat.normalize(quat.between(np.asarray([0, 0, 1]), root_direction))

        self.positions[:, :1] = quat.mul_vec(quat.inv(root_rotation), self.positions[:, :1] - root_position)
        self.rotations[:, :1] = quat.mul(quat.inv(root_rotation), self.rotations[:, :1])

        self.positions = np.concatenate([root_position, self.positions], axis=1)
        self.rotations = np.concatenate([root_rotation, self.rotations], axis=1)
        self.parents = np.concatenate([[-1], self.parents + 1])
        self.bone_names = ['Root'] + self.bone_names

    def fk(self):
        if self.gdirty:
            self._grot, self._gpos = quat.fk(self.rotations, self.positions, self.parents)
            self.gdirty = False
        return self._grot, self._gpos

    def anim_retarget(source_anim: 'Anim', dest_sk: Skeleton, binding: Dict) -> 'Anim':
        """
        :param source_anim: 源动画
        :param dest_sk: 目标骨架
        :param binding: 绑定关系
        :returns: 重定向之后的动画
        """

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
            assert np.asarray(self.parents == obj.parents).all()
            assert self.bone_names == obj.bone_names
            assert self.frame_time == obj.frame_time

            output = Anim(str.format('{}_{}', self.name, obj.name))

            output.parents = self.parents.copy()
            output.bone_names = self.bone_names.copy()
            output.frame_time = self.frame_time

            output.positions = np.concatenate([self.positions, obj.positions])
            output.rotations = np.concatenate([self.rotations, obj.rotations])

            return output

        else:
            raise TypeError(obj)

    def copy(self):
        anim = Anim(self.name)
        anim.positions = self.positions.copy()
        anim.rotations = self.rotations.copy()
        anim.parents = self.parents.copy()
        anim.bone_names = self.bone_names.copy()
        anim.frame_time = self.frame_time
        return anim

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
        orients = np.array([]).reshape((0, 4))
        offsets = np.array([]).reshape((0, 3))
        parents = np.array([], dtype=int)
        frame_time = 0.033333

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

        anim = Anim(os.path.basename(filename))
        anim.rotations = quat.unroll(quat.from_euler(np.radians(rotations), order=rot_order))
        anim.positions = positions * 0.01
        anim.parents = parents
        anim.bone_names = bone_names
        anim.frame_time = frame_time

        return anim

    @staticmethod
    def blend(anim_1: 'Anim', anim_2: 'Anim', alpha):
        alpha = np.asarray(alpha)

        assert alpha.ndim in [0, 1, 2]
        assert np.asarray(anim_1.parents == anim_2.parents).all()
        assert anim_1.bone_names == anim_2.bone_names
        assert anim_1.frame_time == anim_2.frame_time

        output = Anim(str.format('{}_{}_blend', anim_1.name, anim_2.name))

        output.parents = anim_1.parents.copy()
        output.bone_names = anim_1.bone_names.copy()
        output.frame_time = anim_1.frame_time

        output.positions = quat.vec_lerp(anim_1.positions, anim_2.positions, alpha)
        output.rotations = quat.slerp(anim_1.rotations, anim_2.rotations, alpha)

        return output
