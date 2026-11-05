import unittest
import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm
from extremilefed.model import smooth_pinball
from extremilefed.config import ERNNConfig


class GaussianSmoothingTests(unittest.TestCase):
    def test_convolution_integral_and_uniform_bound(self):
        h = .13
        for r in (-1., -.07, 0., .07, 1.):
            for u in (.05, .5, .95):
                v = float(smooth_pinball(torch.tensor([[r]], dtype=torch.float64), torch.tensor([u], dtype=torch.float64), h))
                f = lambda z: max(u*(r-h*z), (u-1)*(r-h*z))*norm.pdf(z)
                reference = quad(f, -np.inf, r/h, epsabs=1e-12)[0] + quad(f, r/h, np.inf, epsabs=1e-12)[0]
                self.assertAlmostEqual(v, reference, places=11)
                gap = v-max(u*r,(u-1)*r)
                self.assertGreaterEqual(gap, -1e-14)
                self.assertLessEqual(gap, h/np.sqrt(2*np.pi)+1e-14)

    def test_analytic_first_second_derivatives_and_zero(self):
        h = .03
        r = torch.tensor([[-.3, -.02, 0., .02, .3]], dtype=torch.float64, requires_grad=True)
        u = torch.tensor([.05, .2, .5, .8, .95], dtype=torch.float64)
        v = smooth_pinball(r,u,h)
        g, = torch.autograd.grad(v.sum(),r,create_graph=True)
        H, = torch.autograd.grad(g.sum(),r)
        np.testing.assert_allclose(g.detach().numpy(), u.numpy()[None,:]-ndtr(-r.detach().numpy()/h), atol=1e-13)
        np.testing.assert_allclose(H.detach().numpy(), norm.pdf(r.detach().numpy()/h)/h, atol=1e-12)
        self.assertAlmostEqual(float(v[0,2]),h/np.sqrt(2*np.pi),places=14)

    def test_extreme_residuals_and_kernel_identity(self):
        for dtype in (torch.float32, torch.float64):
            r = torch.tensor([[-1e4, 0., 1e4]],dtype=dtype,requires_grad=True)
            v = smooth_pinball(r,torch.tensor([.05,.5,.95],dtype=dtype),.03)
            g, = torch.autograd.grad(v.sum(),r)
            self.assertTrue(torch.isfinite(v).all() and torch.isfinite(g).all())
        self.assertEqual(ERNNConfig().smoothing_kernel,"gaussian")
        with self.assertRaises(ValueError):
            ERNNConfig(smoothing_kernel="logistic").validate()
