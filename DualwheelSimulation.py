import matplotlib.pyplot as plt
import numpy as np
import Visualization
from matplotlib.animation import FuncAnimation
from Simulator import SimulationSolver
from scipy import signal

def triangle_hump_generator(position):
    #position vs height
    hump_width = 0.33 # m
    hump_height = 0.2
    return hump_height * 0.5 * (signal.sawtooth(2 * np.pi / hump_width * position, 0.5) + 1.0)

# Input
velocity = 0.5  # m/s
hump_offset = velocity / 2.0

# Car dynamics
m = 2.0  # kg
k = 50.0  # N/m
g = 10  # N / kg
b = 1  # N/ m/s
inertia = 1.0

# Car kinematics
body_width = 0.3 * 2
body_height = 0.2
width_total = 1.0
arm_length = 0.5

def get_wheel_pitch(body_height, wheel_height1, wheel_height2, body_pitch):
    # left
    height_difference1 = wheel_height1 - (body_height + body_width / 2.0 * np.sin(body_pitch))
    # right
    height_difference2 = wheel_height2 - (body_height - body_width / 2.0 * np.sin(body_pitch))
    wheel_pitch1 = np.acos(height_difference1 / arm_length)
    wheel_pitch2 = np.acos(height_difference2 / arm_length)
    return wheel_pitch1, wheel_pitch2

def get_arm_pitchs(wheel_pitch1, wheel_pitch2, body_pitch):
    arm_pitch1 = np.pi / 2 - wheel_pitch1 - body_pitch
    arm_pitch2 = np.pi / 2 - wheel_pitch2 + body_pitch
    return arm_pitch1, arm_pitch2


class DualwheelSimulation(SimulationSolver):
    def __init__(self, init_state):
        super().__init__(init_state)

    def get_inputs_at_t(self, t):
        return {
            "height1": triangle_hump_generator(velocity * t),
            "height2": triangle_hump_generator(velocity * t - hump_offset),
        }

    def get_state_change(self, t, state):
        inputs = self.get_inputs_at_t(t)

        # First get arm pitch. This comes from the fact that the arm is constrained to a circle
        wheel_pitch1, wheel_pitch2 = get_wheel_pitch(body_height=state["body_height"], wheel_height1=inputs["height1"],
                                                     wheel_height2=inputs["height2"], body_pitch=state["body_pitch"])
        arm_pitch1, arm_pitch2 = get_arm_pitchs(wheel_pitch1=wheel_pitch1,
                                                wheel_pitch2=wheel_pitch2,
                                                body_pitch=state["body_pitch"])

        # Then compute torque
        torque1 = k * arm_pitch1 - b * state["body_velocity"] - b * state["body_rot_velocity"]
        torque2 = k * arm_pitch2 - b * state["body_velocity"] + b * state["body_rot_velocity"]

        # Compute normal forces on wheels
        normal1 = torque1 / (arm_length * np.sin(wheel_pitch1))
        normal2 = torque2 / (arm_length * np.sin(wheel_pitch2))

        # Then compute acceleration
        acceleration = (normal1 + normal2) / m - g

        # Then compute inertial acceleration
        rot_accel = (torque1 - torque2 + (normal1 - normal2) * body_width / 2.0 * np.cos(state["body_pitch"])) / inertia

        d_state = {
            "body_height": state["body_velocity"],
            "body_velocity": acceleration,
            "body_pitch": state["body_rot_velocity"],
            "body_rot_velocity": rot_accel
        }
        return d_state

    def get_observations_at_t(self, t, state):
        inputs = self.get_inputs_at_t(t)
        wheel_pitch1, wheel_pitch2 = get_wheel_pitch(body_height=state["body_height"], wheel_height1=inputs["height1"],
                                                     wheel_height2=inputs["height2"], body_pitch=state["body_pitch"])
        # arm_pitch1, arm_pitch2 = get_arm_pitchs(wheel_pitch1=wheel_pitch1,
        #                                         wheel_pitch2=wheel_pitch2,
        #                                         body_pitch=state["body_pitch"])

        observations = {
            "wheel_pitch1": wheel_pitch1,
            "wheel_pitch2": wheel_pitch2,
            "wheel_height1": inputs["height1"],
            "wheel_height2": inputs["height2"],
        }
        return observations


def draw_body(state, ax):
    # draw wheel height
    wheel_line = Visualization.LineWrapper(start=np.array([0.0, 0]), stop=np.array([0.5, 0]))
    wheel_line.translate(np.array([-0.25, 0]))
    wheel_line.translate(np.array([0.9, state["wheel_height1"]]))
    out1 = wheel_line.draw(ax, color='r')

    wheel_line = Visualization.LineWrapper(start=np.array([0.0, 0]), stop=np.array([0.5, 0]))
    wheel_line.translate(np.array([-0.25, 0]))
    wheel_line.translate(np.array([-0.9, state["wheel_height2"]]))
    out2 = wheel_line.draw(ax, color='r')

    # first draw center
    default_box = Visualization.get_box(body_width, body_height)
    default_box.translate(np.array([-body_width / 2.0, 0]))
    default_box.rotate(state["body_pitch"])
    default_box.translate(np.array([0, state["body_height"]]))
    out3 = default_box.draw(ax)

    # Then draw arm
    # right
    default_arm = Visualization.LineWrapper(start=np.array([0, 0]), stop=np.array([arm_length, 0]))
    default_arm.rotate(np.pi/2 - state["wheel_pitch1"])
    default_arm.translate(default_box.points[3])
    out4 = default_arm.draw(ax)
    # left
    default_arm = Visualization.LineWrapper(start=np.array([0, 0]), stop=np.array([arm_length, 0]))
    default_arm.rotate(np.pi - (np.pi/2 - state["wheel_pitch2"]))
    default_arm.translate(default_box.points[0])
    out5 = default_arm.draw(ax)

    return out1, out2, out3, out4, out5


def reset_ax(ax):
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.25, 0.75)
    ax.set_aspect('equal', adjustable='box')

def init():
    reset_ax(ax)
    return draw_body(init_state, ax)

def update(frame):
    # Grab the state at the appropriate timestep
    step = frame
    state = combined_histories[step]
    ax.clear()
    reset_ax(ax)
    return draw_body(state, ax)


if __name__=="__main__":
    init_state = {
        "body_height": 0.0,
        "body_velocity": 0.0,
        "body_pitch": 0.0,
        "body_rot_velocity": 0.0
    }

    solver = DualwheelSimulation(init_state)
    histories, timesteps = solver.solve(dt=0.01, final_time=20)
    solver.plot_histories(histories, timesteps)


    # now animate
    init_state.update({
        "wheel_pitch1": 0.0,
        "wheel_height1": 0.0,
        "wheel_pitch2": 0.0,
        "wheel_height2": 0.0
    })
    combined_histories = solver.combined_histories
    fig, ax = plt.subplots()
    num_steps = len(timesteps)
    ani = FuncAnimation(fig, update, frames=np.arange(0, num_steps), interval=0.2,
                        init_func=init, blit=True)
    plt.show()