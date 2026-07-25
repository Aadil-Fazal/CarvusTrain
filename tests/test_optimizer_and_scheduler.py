"""Unit tests for optimizers and learning rate schedulers."""

import unittest

from carvustrain.optimizer import SGD, Adam, AdamW
from carvustrain.scheduler import CosineAnnealingLR, LinearWarmupLR, StepLR


class TestOptimizerAndScheduler(unittest.TestCase):

    def test_optimizers(self):
        opt = AdamW(lr=1e-3, weight_decay=0.01)
        opt.step()
        self.assertEqual(opt.step_count, 1)

    def test_schedulers(self):
        opt = Adam(lr=1e-2)
        sched = StepLR(opt, step_size=2, gamma=0.5)

        lr0 = opt.lr
        sched.step()
        sched.step()
        self.assertEqual(opt.lr, 1e-2 * 0.5)

    def test_warmup_scheduler(self):
        opt = Adam(lr=1e-2)
        sched = LinearWarmupLR(opt, warmup_steps=10)
        sched.step()
        self.assertLess(opt.lr, 1e-2)


if __name__ == "__main__":
    unittest.main()
