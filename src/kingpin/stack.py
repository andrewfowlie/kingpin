"""
Context managers that overwrite text
====================================
"""

from ezcurses import Cursed


def get_screen_size():
    """
    Dimensions of screen from curses
    """
    with Cursed() as scr:
        return scr.max_size()


class Stack:
    """
    Stack windows that overwrite text
    """

    def __init__(self, rel_orig=0.):
        """
        :param rel_orig:  Relative height of window
        """
        self.rel_orig = rel_orig

    def __call__(self, rel_size):
        """
        Create context manager
        """
        window = Window(rel_orig=(0., self.rel_orig), rel_size=(1., rel_size))
        self.rel_orig += rel_size
        return window


class Window(Cursed):
    """
    Make windows that overwrite text
    """

    def __init__(self, rel_orig=None, rel_size=None):
        """
        :param rel_orig: Origin in relative screen co-ordinates
        :param rel_size: Size in relative screen co-ordinates
        """
        self.rel_size = rel_size if rel_size is not None else (1, 1)
        self.rel_orig = rel_orig if rel_orig is not None else (0, 0)
        self.screen = None
        super().__init__()

    @property
    def size(self):
        """
        :return: Size in terms of current screen co-ordinates
        """
        return self.screen_coords(self.rel_size)

    @property
    def orig(self):
        """
        :return: Origin in terms of current screen co-ordinates
        """
        return self.screen_coords(self.rel_orig)

    def screen_coords(self, fraction):
        """
        Convert relative to screen co-ordinates
        """
        size = self.screen.max_size()
        return (fraction[0] * size[0], fraction[1] * size[1])

    def window_coords(self, fraction):
        """
        Convert relative to window co-ordinates
        """
        return (fraction[0] * self.size[0], fraction[1] * self.size[1])

    def write(self, string, color=None, rel_orig=None):
        """
        Write on window and refresh
        """
        window = self.screen.new_win(orig=self.orig, size=self.size)
        color = color if color is not None else ('white', 'black')
        orig = self.window_coords(rel_orig) if rel_orig is not None else (2, 1)

        window.clear()
        window.border()

        for i, line in enumerate(string.split("\n")):
            window.write(line, (orig[0], i + orig[1]), color=color)

        window.refresh()

    def __enter__(self):
        """
        Context manager for window
        """
        self.screen = super().__enter__()
        return self
