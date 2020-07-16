import numpy as np
import matplotlib.pyplot as plt
import warnings
import pathos
import tqdm

warnings.filterwarnings("ignore")


class DelayLung:
    def __init__(self, **kwargs):
        # dynamics hyperparameters
        self.params = {
            'min_volume': 1.5,
            'R_lung': 10,
            'C_lung': 10,
            'delay': 15,
            'inertia': 0.995,
            'dt': 0.003,
            'control_gain': 0.02
        }

        if kwargs is not None:
            for p in kwargs:
                self.params[p] = kwargs[p]

        # reset states
        self.reset()

    def reset(self):
        self.volume = self.params['min_volume']
        self.pipe_pressure = 0

        self.controls_in, self.controls_out = [], []
        self.volumes, self.pressures = [], []
        self.compute_aux_states()

    def compute_aux_states(self):
        # compute all other state vars, which are just functions of volume
        r = (3 * self.volume / (4 * np.pi)) ** (1 / 3)
        r0 = (3 * self.params['min_volume'] / (4 * np.pi)) ** (1 / 3)
        self.lung_pressure = self.params['C_lung'] * (1 - (r0 / r) ** 6) / (r0 ** 2 * r)

        if len(self.controls_in) < self.params['delay']:
            self.pipe_impulse = 0
            self.peep = 0
        else:
            self.pipe_impulse = self.params['control_gain'] * self.controls_in[-self.params['delay']]
            self.peep = self.controls_out[-self.params['delay']]

        self.pipe_pressure = self.params['inertia'] * self.pipe_pressure + self.pipe_impulse
        self.pressure = max(0, self.pipe_pressure - self.lung_pressure)

        if self.peep:
            self.pipe_pressure *= 0.995

    def step(self, u_in, u_out):
        # 2-dimensional action per timestep
        u_in = max(0, u_in)
        self.controls_in.append(u_in)
        self.controls_out.append(u_out)

        flow = self.pressure / self.params['R_lung']

        # update by flow rate
        self.volume += flow * self.params['dt']
        self.volume = max(self.volume, self.params['min_volume'])

        # compute and record state
        self.compute_aux_states()
        self.volumes.append(self.volume)
        self.pressures.append(self.pressure)


class BreathWaveform:
    def __init__(self, range, keypoints):
        self.lo, self.hi = range
        self.xp = [0] + keypoints
        self.fp = [self.lo, self.hi, self.hi, self.lo, self.lo]

    def at(self, t):
        return np.interp(t, self.xp, self.fp, period=self.xp[-1])


class PID:
    def __init__(self, K, waveform, dt=0.05, RC=0.5):
        # controller coeffs
        self.K_P, self.K_I, self.K_D = K

        # controller states
        self.P, self.I, self.D = 0, 0, 0

        # timescale hyperparams for I/D timescales
        self.RC = RC
        self.dt = dt
        self.decay = dt / (dt + RC)
        self.waveform = waveform

    def feed(self, state, t):
        # ingests current error, updates controller states, outputs PID control
        err = self.waveform.at(t) - state

        self.I += self.decay * (err - self.I)
        self.D += self.decay * (err - self.P - self.D)
        self.P = err

        return self.K_P * self.P + self.K_I * self.I + self.K_D * self.D

    def copy(self):
        pid_copy = PID([self.K_P, self.K_I, self.K_D], waveform=self.waveform, dt=self.dt, RC=self.RC)
        pid_copy.P, pid_copy.I, pid_copy.D = self.P, self.I, self.D
        return pid_copy


allconf = {
    'metrics': {
        'overshoot_penalty': 100
    },
    'time': {
        'T': 10,
        'dt': 0.003,
    },
    'lung': {
        'min_volume': 1.5,
        'R_lung': 10,
        'C_lung': 10,
        'delay': 15,
        'inertia': 0.995,
        'dt': 0.003,
        'control_gain': 0.02
    },
    'waveform': {
        'lohi': [3, 40],
        'keypoints': [0.3, 1.2, 1.7, 3]
    }
}


def loss(target, value, t, is_active=lambda t: True, allconf=allconf):
    loss_val = abs(target - value)
    if target < value:
        loss_val *= allconf['metrics']['overshoot_penalty']
    return loss_val if is_active(t) else 0


def evaluate(controller, lung, waveform, allconf=allconf):
    T, dt = allconf['time']['T'], allconf['time']['dt']
    tt, loss_values = np.arange(T // dt) * dt, []
    for t in tt:
        err = waveform.at(t) - lung.pressure
        pip = controller.feed(lung.pressure, t)
        peep = int(t % allconf['waveform']['keypoints'][-1] > allconf['waveform']['keypoints'][
            1])  # like example code, use breath phase to control peep
        lung.step(pip, peep)

        is_active = lambda t: 0 <= t % allconf['waveform']['keypoints'][-1] and t % allconf['waveform']['keypoints'][
            -1] <= allconf['waveform']['keypoints'][1]
        loss_values.append(loss(waveform.at(t), lung.pressure, t, is_active))
    return loss_values


def benchmark(controlalg, do_plot=False, verbose=False):
    loss_dict = {}
    for C_lung in [6, 8, 10, 12, 14]:
        for delay in [9, 12, 15, 18, 21]:
            allconf["lung"]["C_lung"], allconf["lung"]["delay"] = C_lung, delay

            lung = DelayLung(**allconf["lung"])
            waveform = BreathWaveform(allconf["waveform"]["lohi"], allconf["waveform"]["keypoints"])
            controller = controlalg(waveform)
            loss_values = evaluate(controller, lung, waveform, allconf)
            mean_loss = np.mean(loss_values)
            if verbose:
                print(f"Loss Value with {delay} delay, {C_lung} C_lung = {mean_loss}")
            loss_dict[(delay, C_lung)] = mean_loss

            if do_plot:
                T, dt = allconf["time"]["T"], allconf["time"]["dt"]
                tt = np.arange(T // dt) * dt
                plt.figure(figsize=(10, 4))
                plt.plot(tt, waveform.at(tt), c="k", ls="--")
                plt.plot(tt, lung.pressures, c="b")
                plt.twinx().plot(tt, lung.controls_in, c="gray")
                plt.title(f"Loss Value with {delay} delay, {C_lung} C_lung = {mean_loss}")
    total_mean_loss = np.mean([l for k, l in loss_dict.items()])
    if verbose:
        print(f"Across {len(loss_dict)} runs, mean error is {total_mean_loss}")
    return total_mean_loss


class LinearForecaster:
    def __init__(self, history_length):
        self.history = np.zeros(history_length)
        self._update_lin_fit()

    def update(self, value):
        self.history[0] = value
        self.history = np.roll(self.history, -1)
        self._update_lin_fit()

    def predict(self, steps_ahead):
        return self.lin_fit(len(self.history) + steps_ahead)

    def _update_lin_fit(self):
        self.lin_fit = np.poly1d(np.polyfit(range(len(self.history)), self.history, 1))


class LookaheadController:
    def __init__(self, forecast_model, base_controller, lookahead_steps, dt):
        self.forecast_model = forecast_model
        self.base_controller = base_controller
        self.lookahead_steps = lookahead_steps
        self.dt = dt

    def feed(self, state, t):
        self.forecast_model.update(state)
        control = self.base_controller.feed(state, t)

        controller_copy = self.base_controller.copy()
        for i in range(1, self.lookahead_steps):
            lookahead_state = self.forecast_model.predict(i)
            if lookahead_state is None:
                raise Exception(t)
            control = controller_copy.feed(lookahead_state, t + i * self.dt)
        return control


def hallucinating_pid_controller(K, waveform, dt=0.05, RC=0.1, back_buffer=3, future_buffer=15):
    forecaster = LinearForecaster(back_buffer)
    pid = PID(K, waveform, dt=dt, RC=RC)
    return LookaheadController(forecaster, pid, future_buffer, dt)


class LearnedBiasController:
    def __init__(self, base_controller, bias_lr=0.01, waveform=None):
        self.base_controller = base_controller
        self.bias_lr = bias_lr
        self.bias = 0
        self.waveform = waveform

    def feed(self, state, t):
        err = self.waveform.at(t) - state
        self.bias = self.bias + np.sign(err) * self.bias_lr
        base_control = self.base_controller.feed(state, t)
        return base_control + self.bias

    def copy(self):
        controller_copy = LearnedBiasController(self.base_controller.copy(), self.bias_lr, waveform=self.waveform)
        controller_copy.bias = self.bias
        return controller_copy


def runner(args):
    """Runner function for process pool

    Args:
        args: A pair where the first element is the function to run and
        the second element is a dictionary of kwargs
    Returns:
        Any: the result of the function in the first element
    """
    return args[0](**args[1])


def hallucinating_bias_pi_controller(p, i, RC, future_buffer, waveform, back_buffer=3):
    forecaster = LinearForecaster(back_buffer)
    dt = allconf['time']['dt']
    pid = PID([p, i, 0], waveform, dt=dt, RC=RC)
    lookahead_controller = LookaheadController(forecaster, pid, future_buffer, dt)
    return LearnedBiasController(lookahead_controller, waveform=waveform)


def get_result(p, i, RC, future_buffer, verbose=False, do_plot=False):
    # from vent_lib import hallucinating_bias_pi_controller, benchmark
    controlalg = lambda waveform: hallucinating_bias_pi_controller(p, i, RC, future_buffer, waveform)
    output = {
        "p": p,
        "i": i,
        "RC": RC,
        "future_buffer": future_buffer,
        "err": benchmark(controlalg, verbose=verbose, do_plot=do_plot)
    }
    tqdm.tqdm.write(str(output), file=None, end="\n", nolock=False)
    return output


def generate_arglist():
    import numpy as np
    for p in np.linspace(0, 5, num=10):
        for i in np.linspace(0, 5, num=10):
            for RC in np.linspace(0.003, 0.06, num=10):
                for future_buffer in [3, 9, 15]:
                    if p + i > 0.0:
                        yield get_result, {"p": p, "i": i, "RC": RC, "future_buffer": future_buffer}


if __name__ == '__main__':
    single = lambda: map(runner, generate_arglist())
    parallel = lambda: pathos.pools.ProcessPool().imap(runner, generate_arglist())
    results = list(tqdm.tqdm(parallel()))
    print(results)
