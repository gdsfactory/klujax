"""Downstream integration test: klujax as a SAX circuit-simulator backend.

`SAX <https://github.com/gdsfactory/sax>`_ (S-matrices with Autograd and XLA) is
the flagship consumer of klujax: it uses the KLU backend by default and calls
``klujax.analyze``, ``klujax.solve_with_symbol`` and ``vmap(klujax.dot)``.

SAX pins ``jax<0.10`` and pulls a large dependency tree, so this module is
skipped unless SAX is importable. Run it in a dedicated environment with::

    scripts / verify_sax.sh  # or: just verify-sax
"""

from __future__ import annotations

import numpy as np
import pytest

sax = pytest.importorskip("sax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from sax.backends import default_backend  # noqa: E402


def _waveguide(length: float = 10.0):
    phase = 1j * length / 10
    return sax.reciprocal(
        {("in0", "out0"): jnp.exp(phase), ("in1", "out1"): jnp.exp(phase)}
    )


def _coupler(coupling: float = 0.5):
    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    return sax.reciprocal(
        {
            ("in0", "out0"): tau,
            ("in0", "out1"): 1j * kappa,
            ("in1", "out0"): 1j * kappa,
            ("in1", "out1"): tau,
        }
    )


def _netlist(coupling: float = 0.3, length1: float = 5.0, length2: float = 7.0):
    return {
        "instances": {
            "wg1": {"component": "waveguide", "settings": {"length": length1}},
            "dc1": {"component": "coupler", "settings": {"coupling": coupling}},
            "wg2": {"component": "waveguide", "settings": {"length": length2}},
        },
        "connections": {"wg1,out0": "dc1,in0", "dc1,out0": "wg2,in0"},
        "ports": {"in": "wg1,in0", "out": "wg2,out0", "refl": "dc1,in1"},
    }


_MODELS = {"waveguide": _waveguide, "coupler": _coupler}


def _circuit(backend: str, **settings):
    return sax.circuit(netlist=_netlist(**settings), models=_MODELS, backend=backend)[0]


def _max_diff(a: dict, b: dict) -> float:
    return max(float(jnp.abs(jnp.asarray(a[k]) - jnp.asarray(b[k]))) for k in a)


def test_klu_is_the_default_backend():
    # If klujax imports, sax selects klu by default.
    assert default_backend == "klu"


def test_klu_matches_reference_backend():
    s_klu = _circuit("klu")(wl=1.55)
    s_ref = _circuit("filipsson_gunnar")(wl=1.55)
    assert _max_diff(s_klu, s_ref) < 1e-12


def test_klu_jit_matches_eager():
    c = _circuit("klu")
    s_eager = c(wl=1.55)
    s_jit = jax.jit(c)(wl=1.55)
    assert _max_diff(s_eager, s_jit) < 1e-12


def test_klu_grad_matches_reference_backend():
    def loss(backend, coupling):
        c = _circuit(backend, coupling=coupling)
        return jnp.real(c(wl=1.55)[("out", "in")])

    g_klu = jax.grad(lambda x: loss("klu", x))(0.3)
    g_ref = jax.grad(lambda x: loss("filipsson_gunnar", x))(0.3)
    assert float(jnp.abs(g_klu - g_ref)) < 1e-10


def test_klu_vmap_over_wavelength():
    wl = jnp.linspace(1.5, 1.6, 11)
    s = jax.vmap(_circuit("klu"))(wl=wl)
    assert np.asarray(s[("out", "in")]).shape == (len(wl),)
