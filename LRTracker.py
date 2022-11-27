import numpy as np

class LinearLRTracker:
    def __init__(self, start_lr=1e-3, end_lr=1., num_steps=100, break_loss=None, stop_on_nan_or_inf=True):
        self.current_lr = start_lr
        self.steps = np.linspace(start_lr, end_lr, num_steps)
        self.step_iter = iter(self.steps)
        self.break_loss = break_loss
        self.stop_on_nan_or_inf = stop_on_nan_or_inf
        self.losses = []
        self.num_steps = num_steps

    def lr_step(self) -> float:
        return next(self.step_iter)

    def record_loss(self, loss: float) -> None:
        if self.stop_on_nan_or_inf and np.isnan(loss): raise StopIteration("Got nan!")
        if self.stop_on_nan_or_inf and not np.isfinite(loss): raise StopIteration("Got inf!")
        if self.break_loss is not None and loss > self.break_loss: raise StopIteration("Loss too high!")
        self.losses.append(loss)

    def findLR(self):
        return self.steps[np.argmin(np.diff(self.losses))]
