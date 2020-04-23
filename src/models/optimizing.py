import tensorflow as tf

class WarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, max_lr, warmup_steps):
        super(WarmupScheduler, self).__init__()
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return tf.minimum(self.max_lr, (step/self.warmup_steps) * self.max_lr)
