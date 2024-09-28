from Simulator import SimulationSolver
from scipy import signal
import numpy as np

def triangle_hump_generator(position):
    #position vs height
    hump_width = 0.33 # m
    hump_height = 0.2
    return hump_height * 0.5 * (signal.sawtooth(2 * np.pi / hump_width * position, 0.5) + 1.0)


velocity = 0.1  # m/s
car_length = 0.6

m = 1.5  # kg
k = 50.0  # N/m
g = 10  # N / kg
b = 1  # N/ m/s

class PlanarSimulation(SimulationSolver):
    def __init__(self, init_state):
        super().__init__(init_state)

    def get_inputs_at_t(self, t):
        return {
            "x_f": triangle_hump_generator(velocity * t),
            "x_b": triangle_hump_generator(velocity * t - car_length)
        }

    def get_state_change(self, t, state):
        inputs = self.get_inputs_at_t(t)
        d_state = {
            "x_f": state["v_f"],
            "v_f": (-k / m * (state["x_f"] - inputs["x_f"]) - b * state["v_f"] / m - g),
            "x_b": state["v_b"],
            "v_b": (-k / m * (state["x_b"] - inputs["x_b"]) - b * state["v_b"] / m - g),
        }
        return d_state

    def get_observations_at_t(self, t, state):
        inputs = self.get_inputs_at_t(t)
        observations = {
            "h": (state["x_f"] + state["x_b"] ) / 2.0,
            "input_f": inputs["x_f"],
            "input_b": inputs["x_b"]
        }
        return observations

if __name__ == '__main__':
    init_state = {
        "x_f": 0,
        "v_f": 0,
        "x_b": 0,
        "v_b": 0
    }

    solver = PlanarSimulation(init_state)
    histories, timesteps = solver.solve(dt=0.01, final_time=20)
    solver.plot_histories(histories, timesteps)

