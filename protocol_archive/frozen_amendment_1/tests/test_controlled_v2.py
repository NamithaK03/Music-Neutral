import unittest
import numpy as np
from experiments.controlled_study_v2 import scenario, evaluate, interval
from sonar.controlled_study import scenarios, evaluate as original_evaluate

class ControlledExtensionTests(unittest.TestCase):
    def test_original_stream_and_relabel_equivalence(self):
        old = scenarios(2026,.5,0,n=1000)
        new = scenario(2026,(.5,.5),0)
        for i in (0,1,2,4,5):
            np.testing.assert_array_equal(old[i],new[i])
        for a,b in zip(old[3],new[3]):
            np.testing.assert_array_equal(a,b)
        # No policy evaluation on planned scenarios before the frozen execution.

    def test_hand_constructed_metrics(self):
        a = np.tile(np.eye(4),(20,1))
        genres = np.repeat(np.arange(4),20)
        truth = np.array([0,1])
        p = np.full((2,4),.25)
        completion = np.full((2,80),.5)
        uniforms = np.zeros((2,80))
        data = a,genres,truth,[p,p],completion,uniforms
        for handling in ('ungated','relabel','abstain','downweight'):
            out = evaluate(data,'text',handling,False)
            self.assertAlmostEqual(out['ndcg_at_10'],1)
            self.assertEqual(out['skip_rate'],0)
        self.assertEqual(evaluate(data,'text','abstain',False)['abstention_rate'],1)
        self.assertIsNone(evaluate(data,'text','abstain',False)['covered_accuracy'])
        self.assertEqual(interval([1.]*10)['ci95'],[1.,1.])
        self.assertIsNone(interval([None]*10)['ci95'])
