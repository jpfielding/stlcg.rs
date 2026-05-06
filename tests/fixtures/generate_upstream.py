#!/usr/bin/env python3
"""Generate golden fixtures from upstream StanfordASL/stlcg.

The script intentionally imports the upstream Python implementation directly
from GitHub so fixture updates are reproducible and auditable. It expects
`torch` and `numpy` to be installed in the active Python environment.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import urllib.request
from pathlib import Path


SOURCE_URL = "https://raw.githubusercontent.com/StanfordASL/stlcg/master/src/stlcg.py"
OUTPUT = Path(__file__).with_name("upstream_stlcg.json")


def load_upstream():
    with urllib.request.urlopen(SOURCE_URL) as response:
        source = response.read()

    temp = Path(tempfile.mkdtemp(prefix="stlcg-upstream-"))
    module_path = temp / "stlcg.py"
    module_path.write_bytes(source)

    # Upstream imports IPython only for notebook/visualization conveniences.
    sys.modules.setdefault("IPython", types.ModuleType("IPython"))

    spec = importlib.util.spec_from_file_location("upstream_stlcg", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # Keep old upstream np.Inf references working with newer NumPy.
    setattr(module.np, "Inf", module.np.inf)
    return module


def tensor_payload(tensor):
    tensor = tensor.detach().cpu()
    return {
        "shape": list(tensor.shape),
        "values": tensor.numpy().astype(float).reshape(-1).tolist(),
    }


def case(name, formula, inputs, **kwargs):
    return {
        "name": name,
        "expected": tensor_payload(formula.robustness_trace(inputs, **kwargs)),
    }


def main():
    stlcg = load_upstream()
    torch = stlcg.torch
    np = stlcg.np

    x = torch.tensor([[[3.0], [-1.0], [2.0], [0.0]]], dtype=torch.float32)
    lt2 = stlcg.LessThan(lhs="x", val=2.0)
    ge2 = stlcg.GreaterThan(lhs="x", val=2.0)
    eq2 = stlcg.Equal(lhs="x", val=2.0)
    ident = stlcg.Identity(name="x")

    cases = [
        case("less_equal_x_le_2", lt2, x),
        case("greater_equal_x_ge_2", ge2, x),
        case("equal_x_eq_2", eq2, x),
        case("not_less_equal_x_le_2", stlcg.Negation(lt2), x),
        case("and_ge2_le2", stlcg.And(ge2, lt2), (x, x)),
        case("or_ge2_le2", stlcg.Or(ge2, lt2), (x, x)),
        case("implies_ge2_le2", stlcg.Implies(ge2, lt2), (x, x)),
        case("always_unbounded_le2", stlcg.Always(lt2), x),
        case("eventually_unbounded_le2", stlcg.Eventually(lt2), x),
        case("always_closed_0_1_le2", stlcg.Always(lt2, interval=[0, 1]), x),
        case("eventually_closed_0_1_le2", stlcg.Eventually(lt2, interval=[0, 1]), x),
        case("always_from_1_le2", stlcg.Always(lt2, interval=[1, np.inf]), x),
        case("eventually_from_1_le2", stlcg.Eventually(lt2, interval=[1, np.inf]), x),
        case("or_smooth_scale_10", stlcg.Or(ge2, lt2), (x, x), scale=10.0),
        case("until_unbounded_ge2_until_le2", stlcg.Until(ge2, lt2), (x, x)),
        case("then_unbounded_ge2_then_le2", stlcg.Then(ge2, lt2), (x, x)),
        case("integral_identity_cumulative", stlcg.Integral1d(ident), x),
        case("integral_identity_window_0_1", stlcg.Integral1d(ident, interval=[0, 1]), x),
    ]

    payload = {
        "source_url": SOURCE_URL,
        "trace": tensor_payload(x),
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
