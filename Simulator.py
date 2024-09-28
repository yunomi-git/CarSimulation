import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod

# goal: wrap rk45 using desciptions
class DictHistory:
    def __init__(self, history_dict=None):
        if history_dict is None:
            self.dict = {}
        else:
            self.dict = history_dict

    def add_dict(self, state):
        for key in state.keys():
            if key not in self.dict:
                self.dict[key] = []

            self.dict[key].append(state[key])

    def get_index(self, i):
        state = {}
        for key in self.dict.keys():
            state[key] = self.dict[key][i]
        return state


class SimulationSolver:
    def __init__(self, init_state):
        self.init_state = init_state
        self.state_names = list(init_state.keys())
        self.num_states = len(self.state_names)

    @abstractmethod
    def get_inputs_at_t(self, t):
        pass

    @abstractmethod
    def get_state_change(self, t, state):
        pass

    @abstractmethod
    def get_observations_at_t(self, t, state):
        pass

    def _generate_dx(self):
        def calc_dx(t, x):
            # first convert state to interpretable
            state = {}
            for i in range(self.num_states):
                name = self.state_names[i]
                state[name] = x[i]

            # now find d_state
            d_state = self.get_state_change(t, state)

            # convert dstate to dx
            dx = self._state_to_tuple(d_state)
            return dx
        return calc_dx

    def _state_to_tuple(self, state):
        x = []
        for i in range(self.num_states):
            name = self.state_names[i]
            x.append(state[name])
        return tuple(x)

    def _tuple_to_state(self, x):
        state = {}
        for i in range(self.num_states):
            name = self.state_names[i]
            state[name] = x[i]
        return state

    def solve_state(self, dt, final_time):
        num_steps = int(final_time / dt)
        timesteps = np.linspace(0, num_steps * dt, num_steps)
        sol = solve_ivp(self._generate_dx(), [0, final_time], self._state_to_tuple(self.init_state), t_eval=timesteps)
        histories = self._tuple_to_state(sol.y)
        timesteps = sol.t
        return histories, timesteps

    def solve(self, dt, final_time):
        histories, timesteps = self.solve_state(dt, final_time)
        state_histories = DictHistory(histories)
        observation_histories = DictHistory()

        for i in range(len(timesteps)):
            state = state_histories.get_index(i)
            time = timesteps[i]
            observation = self.get_observations_at_t(time, state)
            observation_histories.add_dict(observation)

        # Now combine
        histories.update(observation_histories.dict)
        return histories, timesteps

    def plot_histories(self, histories, timesteps):
        state_names = list(histories.keys())
        num_states = len(state_names)
        fig, axs = plt.subplots(num_states, 1)
        for i in range(num_states):
            ax = axs[i]
            state_name = state_names[i]
            ax.plot(timesteps, histories[state_name])
            ax.set_ylabel(state_name)
        plt.show()