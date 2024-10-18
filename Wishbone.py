import numpy as np

import util
import matplotlib.pyplot as plt
from Visualization import DrawWrapper, LineWrapper

def law_cosine_angle(a, b, length):
    return np.acos((a**2 + b**2 - length**2) / (2 * a * b))

def law_cosine_length(a, b, angle):
    return np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(angle))
class Wishbone(DrawWrapper):
    def __init__(self, k, arm_x, arm_y, spring_angle):
        # k as linear N / m
        # mount angle as CCW from y
        # Goal: output torque(angle)
        self.spring_default_l = 0.101
        # k ranges from max: 434 - 887, min: 547-660
        # min_l = 0.075
        self.k = k
        self.arm_default_position = np.array([arm_x, arm_y])
        self.arm_l = np.linalg.norm(self.arm_default_position)
        self.mount_position = (self.spring_default_l *
                               util.get_rotation_matrix_2d(spring_angle) @ np.array([0, 1]))
        self.mount_position += self.arm_default_position
        self.mount_l = np.linalg.norm(self.mount_position)
        # Calculate initial arm angle
        # Coordinates: Wheel is to the right. 00 xy is hinge. 0 angle is x
        self.mount_angle = np.atan2(-self.mount_position[0],
                                    self.mount_position[1])
        self.default_hinge_angle_from_mount = law_cosine_angle(self.arm_l,
                                                               self.mount_l,
                                                               self.spring_default_l)
        self.default_hinge_angle_from_horizontal = self.mount_angle - self.default_hinge_angle_from_mount

        self.draw_hinge_position = np.array([0.0, 0.0])
        self.draw_angle = 0
        self.draw_flip = False
        self.draw_rotation = 0

    def get_torque(self, angle):
        # Torque pushes down
        # First get spring deflection
        hinge_angle_from_mount = self.default_hinge_angle_from_mount - angle
        spring_length = law_cosine_length(self.arm_l, self.mount_l, hinge_angle_from_mount)
        # Now get force magnitude
        force_mag = self.k * (self.spring_default_l - spring_length)

        # Find ending position of the arm
        arm_position = self.get_arm_position(angle)

        force_direction = arm_position - self.mount_position
        force_direction = force_direction / np.linalg.norm(force_direction)
        force = force_mag * force_direction

        arm_input = np.array([arm_position[0], arm_position[1], 0])
        force_input = np.array([force[0], force[1], 0])
        moment = np.linalg.cross(arm_input, force_input)
        moment = moment[2]
        return {
            "moment": moment,
            "force_mag": force_mag,
            "force": force,
            "arm_position": arm_position,
            "shock_length": spring_length
        }

    def get_arm_position(self, angle):
        rotation = util.get_rotation_matrix_2d(angle)
        return rotation @ self.arm_default_position

    def translate(self, translation):
        self.draw_hinge_position += translation

    def set_translation(self, translation):
        self.draw_hinge_position = translation

    def set_draw_angle(self, draw_angle):
        self.draw_angle = draw_angle

    def flip_drawing(self, flip):
        self.draw_flip = flip

    def rotate(self, theta):
        self.draw_rotation = theta

    def draw(self, ax, **kargs):
        # find the spring start and stop
        out = self.get_torque(self.draw_angle)
        arm_position = out["arm_position"]
        mount_position = self.mount_position

        if self.draw_flip:
            arm_position[0] = -arm_position[0]
            mount_position[0] = -mount_position[0]

        line = LineWrapper(start=arm_position,
                           stop=mount_position, width=0.01)
        line.rotate(self.draw_rotation)
        line.translate(self.draw_hinge_position)

        # Something for moment
        moment_mag = out["moment"]
        moment_end = np.array([arm_position[0], arm_position[1] - moment_mag])
        line2 = LineWrapper(start=arm_position,
                           stop=moment_end, width=0.005)
        line2.translate(self.draw_hinge_position)
        # line2.translate(arm_position)
        return line.draw(ax, **kargs), line2.draw(ax, **kargs)

if __name__=="__main__":
    k = 500
    arm_x = 0.06
    arm_y = 0.005
    spring_angle = 0.4
    wishbone = Wishbone(k=k, arm_x=arm_x, arm_y=arm_y, spring_angle=spring_angle)
    print(wishbone.mount_position)

    angles = np.linspace(0, np.pi/3, 1000)
    moments = []
    forces = []
    for angle in angles:
        out = wishbone.get_torque(angle)
        moments.append(out["moment"])
        forces.append(out["force_mag"])

    fig, axs = plt.subplots(2)
    angles *= 180.0/np.pi
    ax = axs[0]
    ax.plot(angles, moments)
    ax.set_xlabel("angle")
    ax.set_ylabel("Moment")
    ax = axs[1]
    ax.plot(angles, forces)
    plt.show()


