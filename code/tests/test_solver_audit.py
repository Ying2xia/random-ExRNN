import unittest
import numpy as np
from extremilefed.solver_audit import audit_step

class RulesTests(unittest.TestCase):
    def test_descent_is_not_stationarity(self):
        r=audit_step(2.,1.,[0.],[.2],[10.])
        self.assertEqual(r['classification'],'finite_step_descent')
        self.assertEqual(r['raw_gradient_norm'],10.)
        self.assertFalse(r['theory_certified'])
        self.assertFalse(r['formal_eligible'])
    def test_boundary_mapping(self):
        r=audit_step(0.,-2.,[0.],[2.],[-1.])
        self.assertEqual(r['projected_gradient_mapping_norm'],0.)
        self.assertEqual(r['raw_gradient_norm'],1.)
        self.assertTrue(r['boundary_active'])
    def test_failure_and_stagnation(self):
        self.assertEqual(audit_step(1.,2.,[0.],[.1],[0.])['classification'],'objective_increase')
        self.assertEqual(audit_step(1.,0.,[0.],[3.],[0.])['classification'],'constraint_violation')
        self.assertEqual(audit_step(1.,np.nan,[0.],[0.],[0.])['classification'],'numerical_failure')
        self.assertEqual(audit_step(1.,1.,[0.],[0.],[0.])['classification'],'no_verified_progress')
    def test_interior_and_unconstrained_mapping(self):
        for radius in (2.,None):
            r=audit_step(2.,1.,[0.],[.1],[.2],radius=radius,mapping_step=.5)
            self.assertAlmostEqual(r['projected_gradient_mapping_norm'],.2)
        with self.assertRaises(ValueError):audit_step(2.,1.,[0.,1.],[.1],[.2])

if __name__=='__main__':unittest.main()
