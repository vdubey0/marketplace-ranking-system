"""Regression tests for real shortages, expansion and fill-first optimization."""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd
from allocation.optimizer import optimize_allocation

spec = importlib.util.spec_from_file_location('simulation', Path(__file__).parents[1] / 'scripts/simulate_cold_start.py')
simulation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simulation)


def pairs(rows):
    return pd.DataFrame(rows, columns=['job_id', 'candidate_id', 'performance_probability', 'eligible'])


class ShortageTests(unittest.TestCase):
    def test_full_pool_shortage_stops_at_initial_k(self):
        jobs = pd.DataFrame({'job_id': ['a', 'b'], 'required_certifications': [[], ['license']]})
        candidates = pd.DataFrame({'certifications': [[], []]})
        capacity, counts = simulation.full_pool_capacity(jobs, candidates)
        self.assertEqual(capacity, 1)
        calls = []
        def score(k):
            calls.append(k)
            return pairs([('a','x',.8,True),('b','x',.9,False)])
        _, chosen, vacancies, _ = simulation.allocate_with_expansion(
            ['a','b'],1,2,score,maximum_fill=capacity,full_eligible_counts=counts)
        self.assertEqual(calls,[1])
        self.assertEqual(len(chosen),1)
        self.assertEqual(vacancies.iloc[0].reason,'no_eligible_candidate')

    def test_full_pool_capacity_detects_shared_supply_bottleneck(self):
        jobs = pd.DataFrame({'job_id':['a','b','c'], 'required_certifications':[['x'],['x'],[]]})
        candidates = pd.DataFrame({'certifications':[['x'],[],[]]})
        capacity, _ = simulation.full_pool_capacity(jobs,candidates)
        self.assertEqual(capacity,2)

    def test_fill_count_before_score(self):
        result = optimize_allocation(pairs([('a','x',.99,True), ('a','y',.01,True), ('b','x',.01,True)]))
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result.performance_probability.sum(), .02)

    def test_hall_shortage_and_zero_eligibility(self):
        result = optimize_allocation(pairs([('a','x',.9,True), ('b','x',.8,True),
                                            ('c','y',.7,True), ('c','z',.6,True), ('d','z',1,False)]))
        self.assertEqual(len(result), 2)
        self.assertEqual(result.candidate_id.nunique(), 2)
        self.assertNotIn('d', result.job_id.tolist())
        self.assertTrue(optimize_allocation(pairs([('a','x',.9,False)])).empty)

    def test_expansion_recovers_missing_candidate(self):
        calls = []
        def score(k):
            calls.append(k)
            return pairs([('a','x',.9,True), ('b','x',.8,True)] +
                         ([('a','y',.7,True)] if k == 2 else []))
        _, chosen, vacancies, _ = simulation.allocate_with_expansion(['a','b'],1,2,score)
        self.assertEqual(calls, [1,2])
        self.assertEqual(len(chosen),2)
        self.assertTrue(vacancies.empty)

    def test_unfillable_pool_stops_and_reports_reason(self):
        _, chosen, vacancies, attempts = simulation.allocate_with_expansion(
            ['a','b'],1,2,lambda k: pairs([('a','x',.8,True),('b','x',.9,False)]))
        self.assertEqual(len(chosen),1)
        self.assertEqual(vacancies.iloc[0].reason, 'no_eligible_candidate')
        self.assertEqual(attempts[-1]['retrieve_k'],2)

if __name__ == '__main__':
    unittest.main()
