"""Behavioral tests for privileged-target diagnosis, with no model or data."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from target_rank_diagnostic import inspect, rank, reasons


class TargetDiagnosisTests(unittest.TestCase):
    def test_existing_target_is_not_rescored(self):
        scores={'a':-1.,'b':-2.,'c':-3.,'target':-4.}
        x=inspect(scores,'target',100.)
        self.assertEqual(x['category'],'covered_below_top3')
        self.assertEqual(x['diagnostic_rank'],4)
        self.assertEqual(scores['target'],-4.)

    def test_missing_target_can_be_recovered(self):
        scores={'a':-1.,'b':-2.,'c':-3.}
        x=inspect(scores,'target',-2.5)
        self.assertEqual(x['diagnostic_rank'],3)
        self.assertEqual(x['category'],'missing_reaches_top3')
        self.assertEqual(x['margin_to_top3'],.5)
        self.assertNotIn('target',scores)
        self.assertIn('baseline:potential_recovery',reasons({'baseline':x}))

    def test_missing_but_scoring_limited(self):
        x=inspect({'a':-1.,'b':-2.,'c':-3.},'target',-8.)
        self.assertEqual(x['category'],'missing_still_below_top3')
        self.assertEqual(reasons({'baseline':x}),[])

    def test_cutoff_excludes_target(self):
        x=inspect({'target':0.,'a':-1.,'b':-2.,'c':-3.},'target',None)
        self.assertEqual(x['third_other_word'],'c')
        self.assertEqual(x['diagnostic_rank'],1)

    def test_ties_have_deterministic_lexical_order(self):
        scores={'d':1.,'b':1.,'a':1.,'c':1.}
        self.assertEqual(rank(scores,'c'),3)
        self.assertEqual(rank(scores,'d'),4)
        self.assertEqual(inspect(scores,'d',None)['category'],'covered_below_top3')

    def test_near_boundary_is_always_checked(self):
        x=inspect({'a':-1.,'b':-2.,'c':-3.,'target':-3.05},'target',None)
        self.assertEqual(reasons({'baseline':x}),['baseline:near_top3'])


if __name__=='__main__':unittest.main()
