from pyray import *
import cffi
import numpy as np

from anim import Anim
from anim_pyray import draw_anim
from raygui_tool import HLayout, gui_curve_1d
import quat


class Viewer(object):
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.title = 'Viewer'
        self.animations = []
        self.curves = []
        self.frame_tags = []

    def add_anim(self, position: np.ndarray, anim: Anim):
        assert position.shape == (3,)
        self.animations.append((position, anim))

    def add_curve(self, curve: np.ndarray, min_value, max_value, start_frame=0):
        assert curve.ndim == 1
        self.curves.append((curve, min_value, max_value, start_frame))

    def add_frame_tag(self, frame: int):
        self.frame_tags.append(frame)

    def show(self):
        """ Show animation viewer """

        assert len(self.animations) != 0

        """ Variables """

        i = 0
        pause = False

        nframes = max([anim.positions.shape[0] for _, anim in self.animations])

        camera_dropdown_edit = False
        camera_active = 0
        camera_follow_h_rot = 45
        camera_follow_v_rot = 30

        camera_target = 0
        target_dropdown_edit = False

        frame_spinner_edit = False

        """ Pointers """

        ffi = cffi.FFI()

        p_i = ffi.new('int *', i)
        p_camera_active = ffi.new('int *', camera_active)
        p_camera_target = ffi.new('int *', camera_target)

        """ Initialization """

        camera = Camera3D(Vector3(3, 2, 3), Vector3(0, 1, 0), Vector3(0, 1, 0), 40, CAMERA_PERSPECTIVE)

        set_target_fps(30)
        set_config_flags(FLAG_MSAA_4X_HINT)

        init_window(self.width, self.height, self.title)

        gui_set_style(DEFAULT, TEXT_SIZE, 24)
        gui_set_style(LABEL, TEXT_ALIGNMENT, TEXT_ALIGN_CENTER)

        """ Main Loop """
        while not window_should_close():

            if not pause:
                i = (i + 1) % nframes

            begin_drawing()
            clear_background(WHITE)

            """ Mouse """

            if is_mouse_button_down(MouseButton.MOUSE_BUTTON_MIDDLE) and camera_active == 1:
                delta = get_mouse_delta()
                camera_follow_h_rot = (camera_follow_h_rot - delta.x / 2) % 360
                camera_follow_v_rot = clamp(camera_follow_v_rot + delta.y / 2, -89, 89)

            """ Camera """

            position, target_anim = self.animations[camera_target]
            gpos = target_anim.fk()[1] + position
            ii = min(i, gpos.shape[0] - 1)

            if camera_active == 0:
                camera.position = Vector3(3, 2, 3)
                camera.target = Vector3(gpos[ii, 0, 0], 1, gpos[ii, 0, 2])
            elif camera_active == 1:
                quat_h_rot = quat.from_angle_axis(np.radians(camera_follow_h_rot), np.array([0, 1, 0]))
                quat_v_rot = quat.from_angle_axis(np.radians(camera_follow_v_rot), np.array([0, 0, 1]))
                delta = quat.mul_vec(quat.mul(quat_h_rot, quat_v_rot), np.array([3, 0, 0]))
                camera.position = Vector3(gpos[ii, 0, 0] + delta[0], 0.85 + delta[1], gpos[ii, 0, 2] + delta[2])
                camera.target = Vector3(gpos[ii, 0, 0], 0.85, gpos[ii, 0, 2])

            """ 3D """

            begin_mode_3d(camera)

            draw_grid(20, 1)

            for position, anim in self.animations:
                draw_anim(Vector3(position[0], position[1], position[2]), anim, i)

            end_mode_3d()

            """ UI """

            header = HLayout(Rectangle(0, 0, self.width, 30))
            header.add_item_absolute(160)
            header.add_item_absolute(120)
            header.add_item_absolute(100)
            header.add_item_padding(240)
            header.end()

            gui_label(header.get_rectangle(0), 'Camera Mode')

            p_camera_active[0] = camera_active
            if gui_dropdown_box(header.get_rectangle(1), 'Look;Follow', p_camera_active, camera_dropdown_edit):
                camera_dropdown_edit = not camera_dropdown_edit
            camera_active = p_camera_active[0]

            gui_label(header.get_rectangle(2), 'Target')

            p_camera_target[0] = camera_target
            anim_names = ';'.join([anim.name for _, anim in self.animations])
            if gui_dropdown_box(header.get_rectangle(3), anim_names, p_camera_target, target_dropdown_edit):
                target_dropdown_edit = not target_dropdown_edit
            camera_target = p_camera_target[0]

            footer = HLayout(Rectangle(0, self.height - 30, self.width, 30))
            footer.add_item_absolute(30)
            footer.add_item_padding(1)
            footer.add_item_absolute(120)
            footer.end()

            pause = gui_toggle(footer.get_rectangle(0), '#132#', pause)
            slider_rect = footer.get_rectangle(1)
            i = int(gui_slider_bar(slider_rect, None, None, i, 0, nframes - 1))

            for frame in self.frame_tags:
                x = slider_rect.x + int(frame / (nframes - 1) * slider_rect.width)
                draw_triangle(Vector2(x, slider_rect.y + 10),
                              Vector2(x + 4, slider_rect.y),
                              Vector2(x - 4, slider_rect.y),
                              DARKGRAY)

            p_i[0] = i
            if gui_spinner(footer.get_rectangle(2), None, p_i, 0, nframes - 1, frame_spinner_edit):
                frame_spinner_edit = not frame_spinner_edit
            i = p_i[0]

            i = int(clamp(i, 0, nframes - 1))

            for curve, minval, maxval, start in self.curves:
                gui_curve_1d(Rectangle(0, self.height - 90, self.width, 60), curve, minval, maxval, i - start)

            end_drawing()

        close_window()
