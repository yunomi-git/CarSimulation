import HumpGenerator
import UniwheelSimulationWishbone
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

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
    "desired_body_height": 0.010,
    # "full_body_width": 0.08255, # 3.25" half-width
    "body_width": 0.020,
    "body_height" : 0.01,
    "arm_length" : 0.100,
    "arm_default_angle": -0.162, #aka angle of 0 spring deflection

    # Wishbone stats
    "k" : 500,
    "arm_x" : 0.06, #0.05
    "arm_y" : -0.015,
    "spring_angle" : 0.6
}


hump_generator = HumpGenerator.generate_profile(height=input_parameters["hump_height"],
                                                width=input_parameters["hump_width"])


def plot_responses(name, min_value, max_value, steps, parameters):
    fig, axs = plt.subplots(2)
    ax = axs[0]
    velocities = np.linspace(0.05, 0.15, 4)
    for v in velocities:
        parameters["velocity"] = v
        inputs, outputs = UniwheelSimulationWishbone.get_response(name, min_value, max_value, steps,
                                                                  parameters,
                                                                  UniwheelSimulationWishbone.get_max_force,
                                                                  hump_generator)
        ax.plot(inputs, outputs, label=str(v) + " m/s")
    ax.set_ylabel("Max Force")
    ax.set_xlabel(name)
    ax.legend()

    ax = axs[1]
    for v in velocities:
        parameters["velocity"] = v
        inputs, outputs = UniwheelSimulationWishbone.get_response(name, min_value, max_value, steps,
                                                                  parameters,
                                                                  UniwheelSimulationWishbone.get_min_height_above_ground,
                                                                  hump_generator)
        ax.plot(inputs, outputs, label=str(v) + " m/s")
    ax.set_ylabel("Min height above ground")
    ax.set_xlabel(name)
    plt.show()

def get_responses(parameters):
    # print("height", UniwheelSimulationWishbone.get_min_height(parameters, hump_generator) - parameters["desired_body_height"])
    print("max_force", UniwheelSimulationWishbone.get_max_force(parameters, hump_generator))

if __name__=="__main__":
    mounting_angle = UniwheelSimulationWishbone.get_arm_mounting_angle_to_reach_default_height(
        parameters["desired_body_height"], parameters)
    print("mounting angle", mounting_angle)
    parameters["arm_default_angle"] = mounting_angle

    print("default")
    # get_responses(parameters)

    # plot_responses("k", 440, 880, 10, parameters)
    # plot_responses("b", 0.01, 0.5, 10, parameters)
    # plot_responses("body_width", 0.02, 0.05, 10, parameters)
    # plot_responses("arm_length", 0.020, 0.100, 10, parameters)

    # plot_responses("arm_x", 0.04, 0.08, 10, parameters)
    # plot_responses("spring_angle", 0, np.pi/3, 10, parameters)

    plot_responses("arm_y", -0.05, 0.05, 10, parameters)


