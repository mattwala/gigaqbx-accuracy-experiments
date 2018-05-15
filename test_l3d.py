import l3d
import numpy as np
import numpy.linalg as la
import sumpy.toys as t
import pytest

from sumpy.kernel import LaplaceKernel


ORDERS = [2, 4, 9]


SCALING = 1 / (4 * np.pi)


def compute_approx_convergence_factor(orders, errors):
    poly = np.polyfit(orders, np.log(errors), deg=1)
    return np.exp(poly[0])


_LSRC = np.array([3., 4., 5.])
_LCTR = np.array([1., 0., 0.])
_LCTR2 = np.array([1., 3., 0.])
_LTGT = np.array([1., 1., 1.])
_LCONV = la.norm(_LTGT - _LCTR) / la.norm(_LSRC - _LCTR)

_MSRC = _LTGT
_MCTR = _LCTR
_MCTR2 = _LCTR2
_MTGT = _LSRC
_MCONV = la.norm(_MSRC - _MCTR) / la.norm(_MTGT - _MCTR)
_MCONV2 = la.norm(_MSRC - _MCTR2) / la.norm(_MTGT - _MCTR2)

_M2LSRC = np.array([-3., 4., 5.])
_M2LCTR = np.array([-2., 5., 3.])
_M2LCTR2 = np.array([1., 0., 2.])
_M2LTGT = np.array([0., 0., -1])
_M2LCONV = la.norm(_M2LTGT - _M2LCTR2) / (la.norm(_M2LCTR2 - _M2LCTR) -
                                          la.norm(_M2LCTR - _M2LSRC))


@pytest.mark.parametrize("src, ctr, tgt, expn_class, expected",
    [(_LSRC, _LCTR, _LTGT, l3d.L3DLocalExpansion, _LCONV),
     (_MSRC, _MCTR, _MTGT, l3d.L3DMultipoleExpansion, _MCONV)])
def test_p2e2p(src, ctr, tgt, expn_class, expected):
    errors = []

    rtol = 1e-2
    if not 0 <= expected < 1 / (1 + rtol):
        raise ValueError()

    pot_actual = SCALING / la.norm(tgt - src)

    for order in ORDERS:
        expn = expn_class(LaplaceKernel(3), order)
        coeffs = expn.coefficients_from_source(ctr - src)
        pot_p2e2p = expn.evaluate(coeffs, tgt - ctr)
        errors.append(np.abs(pot_actual - pot_p2e2p))

    conv_factor = compute_approx_convergence_factor(ORDERS, errors)

    assert conv_factor < expected * (1 + rtol)


@pytest.mark.parametrize("src, ctr, tgt, expn_func, expected",
    [(_LSRC, _LCTR, _LTGT, t.local_expand, _LCONV),
     (_MSRC, _MCTR, _MTGT, t.multipole_expand, _MCONV)])
def test_toy_p2e2p(src, ctr, tgt, expn_func, expected):
    src = src.reshape(3, -1)
    tgt = tgt.reshape(3, -1)

    rtol = 1e-2
    if not 0 <= expected < 1 / (1 + rtol):
        raise ValueError()

    from sumpy.kernel import LaplaceKernel
    ctx = l3d.InterpretedToyContext(
        LaplaceKernel(3), l3d.L3DMultipoleExpansion, l3d.L3DLocalExpansion)

    src_pot = t.PointSources(ctx, src, weights=[1])
    pot_actual = src_pot.eval(tgt)

    errors = []

    for order in ORDERS:
        expn = expn_func(src_pot, ctr, order=order)
        pot_p2e2p = expn.eval(tgt)
        errors.append(np.abs(pot_actual - pot_p2e2p))

    conv_factor = compute_approx_convergence_factor(ORDERS, errors)

    assert conv_factor < expected * (1 + rtol)


@pytest.mark.parametrize(
    "src, ctr, ctr2, tgt, from_expn_class, to_expn_class, expected",
    [(_LSRC, _LCTR, _LCTR2, _LTGT,
      l3d.L3DLocalExpansion, l3d.L3DLocalExpansion, _LCONV),
     (_MSRC, _MCTR, _MCTR2, _MTGT,
      l3d.L3DMultipoleExpansion, l3d.L3DMultipoleExpansion, _MCONV2),
     (_M2LSRC, _M2LCTR, _M2LCTR2, _M2LTGT,
      l3d.L3DMultipoleExpansion, l3d.L3DLocalExpansion, _M2LCONV)])
def test_p2e2e2p(src, ctr, ctr2, tgt, from_expn_class, to_expn_class, expected):
    errors = []

    rtol = 1e-2
    if not 0 <= expected < 1 / (1 + rtol):
        raise ValueError()

    pot_actual = SCALING / la.norm(tgt - src)

    for order in ORDERS:
        expn = from_expn_class(LaplaceKernel(3), order)
        coeffs = expn.coefficients_from_source(ctr - src)
        expn2 = to_expn_class(LaplaceKernel(3), order)
        coeffs2 = expn2.translate_from(expn, coeffs, ctr2 - ctr)
        pot_p2e2e2p = expn2.evaluate(coeffs2, tgt - ctr2)
        errors.append(np.abs(pot_actual - pot_p2e2e2p))

    conv_factor = compute_approx_convergence_factor(ORDERS, errors)

    assert conv_factor < expected * (1 + rtol)


@pytest.mark.parametrize(
    "src, ctr, ctr2, tgt, from_expn_func, to_expn_func, expected",
    [(_LSRC, _LCTR, _LCTR2, _LTGT,
      t.local_expand, t.local_expand, _LCONV),
     (_MSRC, _MCTR, _MCTR2, _MTGT,
      t.multipole_expand, t.multipole_expand, _MCONV2),
     (_M2LSRC, _M2LCTR, _M2LCTR2, _M2LTGT,
      t.multipole_expand, t.local_expand, _M2LCONV)])
def test_toy_p2e2e2p(src, ctr, ctr2, tgt, from_expn_func, to_expn_func, expected):
    src = src.reshape(3, -1)
    tgt = tgt.reshape(3, -1)

    rtol = 1e-2
    if not 0 <= expected < 1 / (1 + rtol):
        raise ValueError()

    from sumpy.kernel import LaplaceKernel
    ctx = l3d.InterpretedToyContext(
        LaplaceKernel(3), l3d.L3DMultipoleExpansion, l3d.L3DLocalExpansion)

    errors = []

    src_pot = t.PointSources(ctx, src, weights=[1])
    pot_actual = src_pot.eval(tgt)

    for order in ORDERS:
        expn = from_expn_func(src_pot, ctr, order=order)
        expn2 = to_expn_func(expn, ctr2, order=order)
        pot_p2e2e2p = expn2.eval(tgt)
        errors.append(np.abs(pot_actual - pot_p2e2e2p))

    conv_factor = compute_approx_convergence_factor(ORDERS, errors)

    assert conv_factor < expected * (1 + rtol)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        import py.test
        py.test.cmdline.main([__file__])
