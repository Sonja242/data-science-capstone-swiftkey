"""Behavior checks for fixed ranking rules and binary calibration."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from ranking_rules import scores,prediction,fit_sigmoid,calibrate,logit,sigmoid,wilson,confidence_metrics
from audit_ranking_study import reconstruct


class RankingRulesTests(unittest.TestCase):
    def setUp(self):
        self.rows=[['a',-1.,-2.,-3.,1,1,-3.,False],['long',-2.,-.1,-2.1,2,4,-4.,True],['word',-3.,-.2,-3.2,1,4,-3.2,False]]

    def test_removing_boundary_can_change_order_without_changing_pool(self):
        self.assertEqual(prediction(self.rows,'baseline')['words'][0],'long')
        self.assertEqual(prediction(self.rows,'no_boundary')['words'][0],'a')
        self.assertEqual(set(scores(self.rows,'baseline')),set(scores(self.rows,'no_boundary')))

    def test_duplicate_path_is_not_counted_twice(self):
        values=scores(self.rows,'two_paths')
        self.assertEqual(values['a'],-3.)
        self.assertGreater(values['long'],-2.1)

    def test_independent_character_adjustment_retains_boundary(self):
        words,share,top,gap=reconstruct(self.rows,'char_sqrt')
        self.assertEqual(words[0],'long')
        self.assertAlmostEqual(top,-1.1)
        self.assertAlmostEqual(share,prediction(self.rows,'char_sqrt')['raw_shortlist_share'])

    def test_two_factor_interaction_formula(self):
        a=scores(self.rows,'token_mean');b=scores(self.rows,'token_mean_no_boundary')
        self.assertAlmostEqual(a['long']-b['long'],-.1)

    def test_no_target_argument_or_pool_injection(self):
        self.assertEqual(len(scores(self.rows,'char_sqrt')),3)
        with self.assertRaises(TypeError):prediction(self.rows,'baseline',target='secret')

    def test_raw_share_is_stable_with_large_negative_scores(self):
        rows=[[w,l-1000,b,t-1000,n,c,a-1000,k] for w,l,b,t,n,c,a,k in self.rows]
        self.assertAlmostEqual(prediction(rows,'baseline')['raw_shortlist_share'],prediction(self.rows,'baseline')['raw_shortlist_share'])

    def test_calibrator_solves_its_penalized_score_equations(self):
        raw=np.linspace(.01,.99,200);y=(np.arange(200)%4==0).astype(float)
        model=fit_sigmoid(raw,y);p=calibrate(raw,model)
        x=np.column_stack((np.ones(len(y)),logit(raw)))
        gradient=x.T@(p-y)+np.array([0.,model['slope']])
        self.assertLess(np.abs(gradient).max(),1e-7)
        self.assertLess(confidence_metrics(y,p)['brier'],confidence_metrics(y,raw)['brier'])

    def test_empty_abstention_group_has_no_accuracy_interval(self):
        self.assertEqual(wilson(0,0),[None,None])
        lower,upper=wilson(2,2)
        self.assertLess(lower,.5)
        self.assertEqual(upper,1.)

    def test_reliability_bins_cover_probability_endpoints(self):
        out=confidence_metrics([0,1],[0.,1.])
        self.assertEqual(sum(x['cases'] for x in out['reliability']),2)
        self.assertLess(out['brier'],1e-20)


if __name__=='__main__':unittest.main()
