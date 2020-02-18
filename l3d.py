"""Laplace 3D expansions with interface to sumpy expansion toys

Formulas are found in [1] and [2].

.. [1] Greengard, Leslie. The rapid evaluation of potential fields in particle
       systems. MIT press, 1988.

.. [2] Beatson, Rick, and Leslie Greengard. "A short course on fast multipole
       methods." Wavelets, multilevel methods and elliptic PDEs 1 (1997): 1-37.
"""

__copyright__ = "Copyright (C) 2018 Matt Wala"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import numpy.linalg as la
import yrecursion

from pytools import memoize_method
from sumpy.kernel import LaplaceKernel
from functools import lru_cache


def c2s(p):
    """Convert Cartesian to spherical coordinates."""
    r = la.norm(p)
    theta = np.arccos(p[2] / r)
    phi = np.arctan2(p[1], p[0])

    return (r, theta, phi)


SCALING = 1 / (4 * np.pi)


def Ys(nmax, theta, phi):
    """Evaluate Y^m_n(theta, phi) for 0<=|m|<=n for all 0<=n<=nmax.
    """

    lgndr_vals = yrecursion.ylgndru(nmax, np.cos(theta))

    t = np.exp(1j * phi)
    expvals = [1]
    for i in range(1 + nmax):
        expvals.append(t * expvals[-1])

    i = 0

    result = np.empty((1 + nmax) ** 2, dtype=np.complex128)

    for n in range(1 + nmax):
        for m in range(-n, n + 1):
            if m < 0:
                expval = np.conj(expvals[-m])
            else:
                expval = expvals[m]
            result[i] = lgndr_vals[n, np.abs(m)] * expval
            i += 1

    return result


@lru_cache(maxsize=16384)
def A(m, n):
    assert abs(m) <= n, (m, n)
    from scipy.special import factorial
    return (-1) ** n / np.sqrt(factorial(n - m) * factorial(n + m))


class L3DMultipoleExpansion(object):

    def __init__(self, kernel, order, dtype=np.complex128):
        assert kernel == LaplaceKernel(3)
        self.order = order
        self.dtype = dtype

    @property
    def ncoeffs(self):
        return (self.order + 1) ** 2

    def coefficients_from_source(self, avec):
        coeffs = []

        # avec = center - src
        rho, alpha, beta = c2s(-avec)

        y_vals = Ys(self.order + 1, alpha, beta)

        def my_Y(m, n):
            return y_vals[n * n + (n + m)]
        
        def M(m, n):
            return my_Y(-m, n) * rho ** n

        for n in range(self.order + 1):
            for m in range(-n, n + 1):
                coeffs.append(M(m, n))

        return coeffs

    def evaluate(self, coeffs, bvec):
        r, theta, phi = c2s(bvec)

        def M(m, n):
            assert abs(m) <= n
            return coeffs[n * n + (n + m)]

        val = 0

        y_vals = Ys(self.order + 1, theta, phi)

        def my_Y(m, n):
            return y_vals[n * n + (n + m)]

        for n in range(self.order + 1):
            for m in range(-n, n + 1):
                val += M(m, n) / r ** (n + 1) * my_Y(m, n)

        return val * SCALING

    def translate_from(self, src_expansion, src_coeffs, dvec):
        assert src_expansion.order == self.order

        def O_(m, n):
            assert abs(m) <= abs(n)
            return src_coeffs[n * n + (n + m)]

        def idx(m, n):
            assert abs(m) <= abs(n)
            assert n <= self.order
            return n * n + (n + m)

        def J(m, n):
            if m * n < 0:
                return (-1) ** min(abs(m), abs(n))
            return 1

        rho, alpha, beta = c2s(-dvec)
        
        y_vals = Ys(self.order + 1, alpha, beta)

        def my_Y(m, n):
            return y_vals[n * n + (n + m)]

        coeffs = np.zeros((self.order + 1) ** 2, dtype=np.complex128)

        # The formula for multipole-to-multipole translation as found in
        # Greengard's thesis is incorrect. The simplest way to see this is that
        # if you plug in n = j into the formula, you get coefficients that are
        # "out of bounds".
        for n in range(self.order + 1):
            for m in range(-n, n + 1):
                for j in range(self.order + 1 - n):
                    for k in range(-j, j + 1):
                        coeffs[idx(k + m, n + j)] += O_(m, n) * (
                            J(m, k) *
                            A(k, j) *
                            A(m, n) *
                            rho ** j *
                            my_Y(-k, j)) / A(k + m, j + n)

        return coeffs


class L3DLocalExpansion(object):

    def __init__(self, kernel, order, dtype=np.complex128):
        assert kernel == LaplaceKernel(3)
        self.order = order
        self.dtype = dtype

    @property
    def ncoeffs(self):
        return (self.order + 1) ** 2

    def coefficients_from_source(self, avec):
        # avec = center - src
        rho, alpha, beta = c2s(-avec)

        y_vals = Ys(self.order + 1, alpha, beta)

        def my_Y(m, n):
            return y_vals[n * n + (n + m)]

        def L(m, n):
            assert abs(m) <= n
            return my_Y(-m, n) / rho ** (n + 1)

        coeffs = []

        for n in range(self.order + 1):
            for m in range(-n, n + 1):
                coeffs.append(L(m, n))

        return coeffs

    def evaluate(self, coeffs, bvec):
        if la.norm(bvec) == 0:
            return coeffs[0] * SCALING

        r, theta, phi = c2s(bvec)

        def L(m, n):
            assert abs(m) <= n
            return coeffs[n * n + (n + m)]

        y_vals = Ys(self.order + 1, theta, phi)

        def my_Y(m, n):
            return y_vals[n * n + (n + m)]

        val = 0

        for n in range(self.order + 1):
            sub_val = 0
            for m in range(-n, n + 1):
                sub_val += L(m, n) * my_Y(m, n)
            sub_val *= r ** n
            val += sub_val

        return val * SCALING

    def translate_from(self, src_expansion, src_coeffs, dvec):
        if isinstance(src_expansion, L3DLocalExpansion):
            return self._l2l(src_expansion, src_coeffs, dvec)
        if isinstance(src_expansion, L3DMultipoleExpansion):
            return self._m2l(src_expansion, src_coeffs, dvec)
        raise NotImplementedError("Unknown expansion class")

    def _l2l(self, src_expansion, src_coeffs, dvec):
        p = src_expansion.order

        if la.norm(dvec) == 0:
            result = np.zeros(self.ncoeffs, dtype=np.complex128)
            l = min(self.ncoeffs, src_expansion.ncoeffs)
            result[:l] = src_coeffs[:l]
            return result

        rho, alpha, beta = c2s(-dvec)

        def O_(m, n):
            assert abs(m) <= n <= p
            return src_coeffs[(n * n) + (m + n)]

        coeffs = []

        y_vals = Ys(p + 1, alpha, beta)

        def my_Y(m, n):
            return y_vals[n * n + (n + m)]

        def L(k, j):
            val = 0
            for n in range(j, p + 1):
                for m in range(j - n + k, n - j + k + 1):
                    val += O_(m, n) * (
                        1j ** (abs(m) - abs(m - k) - abs(k)) *
                        A(m - k, n - j) *
                        A(k, j) *
                        my_Y(m - k, n - j) *
                        rho ** (n - j)) / ((-1) ** (n + j) * A(m, n))
            return val

        for j in range(self.order + 1):
            for k in range(-j, j + 1):
                coeffs.append(L(k, j))

        return coeffs

    def _m2l(self, src_expansion, src_coeffs, dvec):
        p = src_expansion.order

        rho, alpha, beta = c2s(-dvec)

        def J(m, n):
            if m * n < 0:
                return (-1) ** min(abs(m), abs(n))
            return 1

        def O_(m, n):
            assert abs(m) <= n <= p
            return src_coeffs[(n * n) + (m + n)]
        
        y_vals = Ys(self.order + p, alpha, beta)

        def my_Y(m, n):
            return y_vals[n * n + (n + m)]

        def L(k, j):
            val = 0
            for n in range(0, p + 1):
                r = rho ** (j + n + 1)
                for m in range(-n, n + 1):
                    val += O_(m, n) * (
                        1j ** (abs(k - m) - abs(k) - abs(m)) *
                        A(m, n) *
                        A(k, j) *
                        my_Y(m - k, j + n)) / ((-1) ** n * A(m - k, j + n) * r)
            return val

        coeffs = []
        for j in range(self.order + 1):
            for k in range(-j, j + 1):
                coeffs.append(L(k, j))

        return coeffs


class InterpretedToyContext(object):
    """Has the same interface as ToyContext. Uses expansion classes that are
    directly interpreted as opposed to compiled.
    """
    def __init__(self, kernel, mpole_expn_class, local_expn_class):
        self.kernel = kernel
        self.mpole_expn_class = mpole_expn_class
        self.local_expn_class = local_expn_class

    @property
    def queue(self):
        return None

    @property
    def extra_source_and_kernel_kwargs(self):
        return {}

    @property
    def extra_kernel_kwargs(self):
        return {}

    @memoize_method
    def get_p2p(self):
        return _P2P(self.kernel)

    @memoize_method
    def get_p2l(self, order):
        return _P2EFromSingleBox(self.local_expn_class(self.kernel, order))

    @memoize_method
    def get_p2m(self, order):
        return _P2EFromSingleBox(self.mpole_expn_class(self.kernel, order))

    @memoize_method
    def get_l2p(self, order):
        return _E2PFromSingleBox(self.local_expn_class(self.kernel, order))

    @memoize_method
    def get_m2p(self, order):
        return _E2PFromSingleBox(self.mpole_expn_class(self.kernel, order))

    @memoize_method
    def get_l2l(self, from_order, to_order):
        return _E2EFromCSR(self.local_expn_class(self.kernel, from_order),
                           self.local_expn_class(self.kernel, to_order))

    @memoize_method
    def get_m2l(self, from_order, to_order):
        return _E2EFromCSR(self.mpole_expn_class(self.kernel, from_order),
                           self.local_expn_class(self.kernel, to_order))

    @memoize_method
    def get_m2m(self, from_order, to_order):
        return _E2EFromCSR(self.mpole_expn_class(self.kernel, from_order),
                           self.mpole_expn_class(self.kernel, to_order))


class _P2P(object):

    def __init__(self, kernel):
        assert kernel == LaplaceKernel(3)

    def __call__(self, queue, targets, sources, weights, **kwargs):
        result = []
        for target in targets.T:
            val = 0
            for source, weight in zip(sources.T, weights[0]):
                val += weight / la.norm(source - target)
            result.append(val)
        return None, (SCALING * np.array(result),)


class _P2EFromSingleBox(object):

    def __init__(self, expansion):
        self.expansion = expansion

    def __call__(self, queue, source_boxes, box_source_starts,
                 box_source_counts_nonchild, centers, sources, strengths,
                 rscale, nboxes, tgt_base_ibox, **kwargs):
        result = []
        for box in source_boxes:
            coeffs = np.zeros(self.expansion.ncoeffs, self.expansion.dtype)
            sources_start = box_source_starts[box]
            sources_end = sources_start + box_source_counts_nonchild[box]
            for source, strength in zip(sources[:, sources_start:sources_end].T,
                                        strengths[sources_start:sources_end]):
                avec = centers[:, box] - source
                coeffs[:] += strength * self.expansion.coefficients_from_source(avec)
            result.append(coeffs)
        return None, (result,)


class _E2PFromSingleBox(object):

    def __init__(self, expansion):
        self.expansion = expansion

    def __call__(self, queue, src_expansions, src_base_ibox, target_boxes,
                 box_target_starts, box_target_counts_nonchild, centers, rscale,
                 targets, **kwargs):
        result = []
        for box in target_boxes:
            targets_start = box_target_starts[box]
            targets_end = targets_start + box_target_counts_nonchild[box]
            for target in targets[:, targets_start:targets_end].T:
                dvec = target - centers[:, box]
                result.append(self.expansion.evaluate(src_expansions[box], dvec))
        return None, (np.array(result),)


class _E2EFromCSR(object):

    def __init__(self, from_expansion, to_expansion):
        self.from_expansion = from_expansion
        self.to_expansion = to_expansion

    def __call__(self, queue, src_expansions, src_base_ibox, tgt_base_ibox,
                 ntgt_level_boxes, target_boxes, src_box_starts, src_box_lists,
                 centers, src_rscale, tgt_rscale, **kwargs):
        result = {}
        for ibox, box in enumerate(target_boxes):
            coeffs = np.zeros(self.to_expansion.ncoeffs, self.to_expansion.dtype)
            tgt_center = centers[:, box]

            source_box_list_start = src_box_starts[ibox]
            source_box_list_end = src_box_starts[ibox + 1]
            source_boxes = src_box_lists[source_box_list_start:source_box_list_end]
            for src_box in source_boxes:
                src_center = centers[:, src_box]
                bvec = tgt_center - src_center
                coeffs[:] += (
                    self.to_expansion.translate_from(self.from_expansion,
                                                     src_expansions[src_box], bvec))
            result[box] = np.array(coeffs)

        return None, (result,)
