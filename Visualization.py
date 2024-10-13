import numpy as np
from matplotlib.patches import Polygon, Circle

from util import get_rotation_matrix_2d
from abc import ABC, abstractmethod

def get_box(width, height):
    return PolygonWrapper(np.array([[0, 0], [0, height], [width, height], [width, 0]]))

class DrawWrapper(ABC):
    @abstractmethod
    def translate(self, translation):
        pass

    @abstractmethod
    def rotate(self, theta):
        pass

    @abstractmethod
    def draw(self, ax, **kargs):
        pass

class CircleWrapper(DrawWrapper):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def translate(self, translation):
        self.center += translation

    def rotate(self, theta):
        pass

    def draw(self, ax, **kargs):
        # convert numpy to polygon
        # poly_list = [tuple(point) for point in self.points]
        return ax.add_patch(Circle(self.center, **kargs))

class PolygonWrapper(DrawWrapper):
    def __init__(self, points=None):
        self.points = points

    def translate(self, translation):
        self.points += translation

    def rotate(self, theta):
        # rotate about 0, 0
        self.points = (get_rotation_matrix_2d(theta) @ self.points.T).T

    def draw(self, ax, **kargs):
        # convert numpy to polygon
        # poly_list = [tuple(point) for point in self.points]
        return ax.add_patch(Polygon(self.points, **kargs))

class LineWrapper(DrawWrapper):
    def __init__(self, start, stop, width=0.02):
        self.start = start
        self.stop = stop
        self.width = width

    def translate(self, translation):
        self.start += translation
        self.stop += translation

    def rotate(self, theta):
        # rotate about 0, 0
        self.start = get_rotation_matrix_2d(theta) @ self.start
        self.stop = get_rotation_matrix_2d(theta) @ self.stop

    def draw(self, ax, **kargs):
        angle = np.atan2(self.stop[1] - self.start[1], self.stop[0] - self.start[0])
        length = np.linalg.norm(self.stop - self.start)
        # Draw a rectangle
        rectangle = get_box(length, self.width)
        rectangle.translate(np.array([0, -self.width/2]))
        rectangle.rotate(angle)
        rectangle.translate(self.start)
        return rectangle.draw(ax, **kargs)
