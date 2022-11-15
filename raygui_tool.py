from pyray import *
import numpy as np


class HLayout(object):
    def __init__(self, rect: Rectangle) -> None:
        self.rect: Rectangle = rect
        self._items = []
        self._ended = False

    def add_item_absolute(self, width: int):
        assert not self._ended
        self._items.append([width, 0])

    def add_item_relative(self, width_percentage: int):
        assert not self._ended
        self._items.append([int(self.rect.width * width_percentage / 100), 1])

    def add_item_padding(self, weight: int):
        assert not self._ended
        self._items.append([0, 2, weight])

    def end(self):
        rest = self.rect.width
        total = 0

        for item in self._items:
            if item[1] == 2:
                total += item[2]
            else:
                rest -= item[0]

        for item in self._items:
            if item[1] == 2:
                item[0] = int(rest * item[2] / total)

        self._ended = True

    def get_rectangle(self, index: int):
        assert self._ended
        return Rectangle(self.rect.x + sum([self._items[i][0] for i in range(index)]),
                         self.rect.y, self._items[index][0], self.rect.height)


def gui_curve_1d(rectangle: Rectangle, curve: np.ndarray, min_value: float, max_value: float, cursor: int):
    assert curve.ndim == 1

    y = np.interp(np.linspace(0, curve.shape[0] - 1, int(rectangle.width)), np.arange(curve.shape[0]), curve)
    y = rectangle.y + ((max_value - y) / (max_value - min_value)) * rectangle.height

    cursor = int(clamp(cursor, 0, curve.shape[0] - 1))
    cursor_x = rectangle.x + cursor / (curve.shape[0] - 1) * rectangle.width
    cursor_y = rectangle.y + ((max_value - curve[cursor]) / (max_value - min_value)) * rectangle.height

    draw_rectangle_rec(rectangle, color_alpha(GRAY, 0.3))

    for i in range(1, y.shape[0]):
        draw_line(int(rectangle.x + i - 1), int(y[i - 1]), int(rectangle.x + i), int(y[i]), BLACK)

    draw_circle(int(cursor_x), int(cursor_y), 5, BLACK)
