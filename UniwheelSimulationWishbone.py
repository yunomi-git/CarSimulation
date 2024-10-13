import matplotlib.pyplot as plt
import numpy as np
import Visualization
from matplotlib.animation import FuncAnimation

import util
from Simulator import SimulationSolver
from scipy import signal
from Wishbone import Wishbone
import HumpGenerator
from tqdm import tqdm

# Input
input_parameters = {
    "hump_width" : 0.07,  # m
    "hump_height" : 0.04, #0.04
}

parameters = {
    # Car dynamics
    "m": 1.5/4,  # kg
    "g" : 10,  # N / kg
    "b" : 0.3,  # N/ m/s
    "velocity": 0.1,  # m/s

    # Car kinematics
    "desired_body_height": 0.02,
    "body_width" : 0.1,
    "body_height" : 0.01,
    "arm_length" : 0.10,
    "arm_default_angle": -0.4155, #aka angle of 0 spring deflection

    # Wishbone stats
    "k" : 500,
    "arm_x" : 0.06, #0.05
    "arm_y" : 0.005,
    "spring_angle" : 0.4
}

hump_generator = HumpGenerator.generate_profile(height=input_parameters["hump_height"],
                                                width=input_parameters["hump_width"])
class UniwheelSimulation(SimulationSolver):
    def __init__(self, init_state, dt, parameters, hump_generator):
        super().__init__(init_state, dt)
        self.m = parameters["m"]  # kg
        self.g = parameters["g"]
        self.b = parameters["b"]

        # Car kinematics
        self.body_width = parameters["body_width"]
        self.body_height = parameters["body_height"]
        self.arm_length = parameters["arm_length"]
        self.arm_default_angle = parameters["arm_default_angle"]

        # Wishbone stats
        self.k = parameters["k"]
        self.arm_x = parameters["arm_x"]
        self.arm_y = parameters["arm_y"]
        self.spring_angle = parameters["spring_angle"]


        self.velocity = parameters["velocity"]

        self.wishbone = Wishbone(k=parameters["k"], arm_x=parameters["arm_x"],
                        arm_y=parameters["arm_y"], spring_angle=parameters["spring_angle"])

        self.hump_generator = hump_generator


    def get_inputs_at_t(self, t):
        return {
            "height": self.hump_generator(self.velocity * t),
        }

    def get_state_change(self, t, state):
        inputs = self.get_inputs_at_t(t)

        # First get arm pitch. This comes from the fact that the arm is constrained to a circle
        height_difference = inputs["height"] - state["body_height"]

        ratio = height_difference / self.arm_length
        ratio = min(1, ratio)
        ratio = max(-1, ratio)

        arm_pitch = np.asin(ratio)

        # Then compute torque
        out = self.wishbone.get_torque(arm_pitch - self.arm_default_angle)
        torque = -out["moment"] - self.b * state["body_velocity"]
        # torque = k * arm_pitch - b * state["body_velocity"]

        # Then compute acceleration
        acceleration = torque / self.body_width / self.m - self.g

        d_state = {
            # "arm_pitch": arm_pitch,
            "body_height": state["body_velocity"],
            "body_velocity": acceleration,
        }
        return d_state

    def get_observations_at_t(self, t, state, prev_state):
        inputs = self.get_inputs_at_t(t)
        height_difference = inputs["height"] - state["body_height"]
        arm_pitch = np.asin(height_difference / self.arm_length)
        body_force = self.m * (state["body_velocity"] - prev_state["body_velocity"]) / self.dt
        observations = {
            "arm_pitch": arm_pitch,
            "wheel_height": inputs["height"],
            "body_force": body_force
        }
        return observations


    def draw_body(self, state, ax):
        # draw wheel height
        wheel_line = Visualization.LineWrapper(start=np.array([0.0, 0]), stop=np.array([0.5, 0]), width=0.01)
        wheel_line.translate(np.array([-0.25, 0]))
        wheel_line.translate(np.array([0.2, state["wheel_height"]]))
        out3 = wheel_line.draw(ax, color='r')

        # first draw center
        default_box = Visualization.get_box(self.body_width, self.body_height)
        default_box.translate(np.array([0, state["body_height"]]))
        out1 = default_box.draw(ax)

        # Then draw arm
        default_arm = Visualization.LineWrapper(start=np.array([0, 0]), stop=np.array([self.arm_length, 0]), width=0.01)
        # rotate
        default_arm.rotate(state["arm_pitch"])
        default_arm.translate(default_box.points[3])
        out2 = default_arm.draw(ax)

        # THen draw linkage
        self.wishbone.set_draw_angle(state["arm_pitch"] - self.arm_default_angle)
        self.wishbone.set_translation(np.array([self.body_width, state["body_height"]]))
        out_w = self.wishbone.draw(ax, color='g')

        return out3, out1, out2, *out_w


def reset_ax(ax):
    ax.set_xlim(0, 0.2)
    ax.set_ylim(-0.05, 0.15)
    ax.set_aspect('equal', adjustable='box')

def init():
    reset_ax(ax)
    return solver.draw_body(init_state, ax)

def update(frame):
    # Grab the state at the appropriate timestep
    step = frame
    state = combined_histories[step]
    ax.clear()
    reset_ax(ax)
    return solver.draw_body(state, ax)


def get_default_height(parameters: dict):
    # set b to be high, and wait for steady state
    init_state = {
        "body_height": 0.0,
        "body_velocity": 0.0,
    }
    parameters = parameters.copy()
    parameters["b"] = 0.5

    hump_generator = HumpGenerator.generate_profile(height=0, width=1)
    solver = UniwheelSimulation(init_state, dt=0.1, parameters=parameters, hump_generator=hump_generator)
    histories, timesteps = solver.solve(final_time=5)
    end_height = histories["body_height"][-1]
    return end_height

def get_arm_mounting_angle_to_reach_default_height(desired_height, parameters: dict):
    min_x = 0.0
    max_x = -np.pi/4
    def function(x):
        parameters["arm_default_angle"] = x
        return get_default_height(parameters) - desired_height
    return util.binary_search(minx=min_x, maxx=max_x, function=function)

def get_min_height_above_ground(parameters: dict, hump_generator):
    init_state = {
        "body_height": parameters["desired_body_height"],
        "body_velocity": 0.0,
    }
    solver = UniwheelSimulation(init_state, dt=0.01, parameters=parameters, hump_generator=hump_generator)
    histories, timesteps = solver.solve(final_time=10)
    height_above_ground = histories["body_height"] - histories["wheel_height"]
    min_height = np.min(height_above_ground[len(timesteps)//2:])
    return min_height

def get_max_force(parameters:dict, hump_generator):
    init_state = {
        "body_height": parameters["desired_body_height"],
        "body_velocity": 0.0,
    }
    solver = UniwheelSimulation(init_state, dt=0.01, parameters=parameters, hump_generator=hump_generator)
    histories, timesteps = solver.solve(final_time=10)
    max_force = histories["body_force"][len(timesteps)//2:]
    max_force = np.max(np.abs(max_force))
    return max_force

def get_response(param_name, min_value, max_value, steps, default_parameters, response_function, *args):
    # response function is f(parameters)
    default_parameters = default_parameters.copy()
    inputs_space = np.linspace(min_value, max_value, steps)
    inputs = []
    outputs = []
    for val in tqdm(inputs_space):
        default_parameters[param_name] = val
        try:
            out = response_function(default_parameters, *args)
            outputs.append(out)
            inputs.append(val)
        except:
            continue
    return inputs, np.array(outputs)


if __name__=="__main__":
    default_height = 0.00
    # mounting_angle = get_arm_mounting_angle_to_reach_default_height(default_height, parameters)
    # print(mounting_angle)
    # parameters["arm_default_angle"] = mounting_angle
    init_state = {
        "body_height": 0,
        "body_velocity": 0.0,
    }
    print("height", get_default_height(parameters))
    print("max_force", get_max_force(parameters, hump_generator))

    solver = UniwheelSimulation(init_state, dt=0.01, parameters=parameters, hump_generator=hump_generator)
    histories, timesteps = solver.solve(final_time=20)

    solver.plot_histories(histories, timesteps)


    # now animate
    init_state.update({
        "arm_pitch": 0.0,
        "wheel_height": 0.0,
        "body_force": 0.0
    })
    combined_histories = solver.combined_histories
    fig, ax = plt.subplots()
    num_steps = len(timesteps)
    ani = FuncAnimation(fig, update, frames=np.arange(0, num_steps), interval=0.2,
                        init_func=init, blit=True)
    plt.show()