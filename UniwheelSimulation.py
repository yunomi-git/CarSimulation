import matplotlib.pyplot as plt
import numpy as np
import Visualization
from matplotlib.animation import FuncAnimation
from Simulator import SimulationSolver
from scipy import signal

def triangle_hump_generator(position):
    #position vs height
    hump_width = 0.33 # m
    hump_height = 0.0
    return hump_height * 0.5 * (signal.sawtooth(2 * np.pi / hump_width * position, 0.5) + 1.0)

# Input
velocity = 0.5  # m/s

# Car dynamics
m = 1.5  # kg
k = 50.0  # N/m
g = 10  # N / kg
b = 1  # N/ m/s

# Car kinematics
body_width = 0.3
body_height = 0.2
width_total = 1.0
arm_length = 0.5


class UniwheelSimulation(SimulationSolver):
    def __init__(self, init_state):
        super().__init__(init_state)

    def get_inputs_at_t(self, t):
        return {
            "height": triangle_hump_generator(velocity * t),
        }

    def get_state_change(self, t, state):
        inputs = self.get_inputs_at_t(t)

        # First get arm pitch. This comes from the fact that the arm is constrained to a circle
        height_difference = inputs["height"] - state["body_height"]
        arm_pitch = np.asin(height_difference / arm_length)

        # Then compute torque
        torque = k * arm_pitch - b * state["body_velocity"]

        # Then compute acceleration
        acceleration = torque / body_width / m - g

        d_state = {
            # "arm_pitch": arm_pitch,
            "body_height": state["body_velocity"],
            "body_velocity": acceleration,
        }
        return d_state

    def get_observations_at_t(self, t, state):
        inputs = self.get_inputs_at_t(t)
        height_difference = inputs["height"] - state["body_height"]
        arm_pitch = np.asin(height_difference / arm_length)
        observations = {
            "arm_pitch": arm_pitch,
            "wheel_height": inputs["height"],
        }
        return observations


def draw_body(state, ax):
    # draw wheel height
    wheel_line = Visualization.LineWrapper(start=np.array([0.0, 0]), stop=np.array([0.5, 0]))
    wheel_line.translate(np.array([-0.25, 0]))
    wheel_line.translate(np.array([0.5, state["wheel_height"]]))
    out3 = wheel_line.draw(ax, color='r')

    # first draw center
    default_box = Visualization.get_box(body_width, body_height)
    default_box.translate(np.array([0, state["body_height"]]))
    out1 = default_box.draw(ax)

    # Then draw arm
    default_arm = Visualization.LineWrapper(start=np.array([0, 0]), stop=np.array([arm_length, 0]))
    # rotate
    default_arm.rotate(state["arm_pitch"])
    default_arm.translate(default_box.points[3])
    out2 = default_arm.draw(ax)



    return out3, out1, out2


def reset_ax(ax):
    ax.set_xlim(0, 1)
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
    }

    solver = UniwheelSimulation(init_state)
    histories, timesteps = solver.solve(dt=0.01, final_time=20)
    solver.plot_histories(histories, timesteps)


    # now animate
    init_state.update({
        "arm_pitch": 0.0,
        "wheel_height": 0.0
    })
    combined_histories = solver.combined_histories
    fig, ax = plt.subplots()
    num_steps = len(timesteps)
    ani = FuncAnimation(fig, update, frames=np.arange(0, num_steps), interval=0.2,
                        init_func=init, blit=True)
    plt.show()