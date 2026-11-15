import unittest
import numpy as np
from sonar.controlled_study import scenarios, evaluate


class ControlledStudyTests(unittest.TestCase):
    def test_policies_do_not_modify_shared_scenarios(self):
        data=scenarios(2026,0.5,0.5,n=30)
        before=data[-1].copy()
        first=evaluate(data,'rescaled',False)
        evaluate(data,'fixed',True)
        self.assertEqual(first,evaluate(data,'rescaled',False))
        np.testing.assert_array_equal(before,data[-1])

    def test_shared_noise_draws_and_valid_metrics(self):
        low=scenarios(2026,0.5,0.0,n=30)
        high=scenarios(2026,1.5,0.5,n=30)
        np.testing.assert_array_equal(low[-1],high[-1])
        np.testing.assert_array_equal(low[2],high[2])
        for value in evaluate(low,'fixed',True).values():
            self.assertTrue(0 <= value <= 1)
