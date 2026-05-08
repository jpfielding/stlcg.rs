# stlcg.rs

[![CI](https://github.com/jpfielding/stlcg.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/jpfielding/stlcg.rs/actions/workflows/ci.yml)

An idiomatic Rust port of the core StanfordASL STLCG robustness evaluator.

The crate uses [Burn](https://burn.dev/) tensors and is generic over Burn
backends. Version 0.1 focuses on core Signal Temporal Logic robustness
evaluation over reverse-time traces shaped `[batch, time, dim]`.

This is not an official StanfordASL project. It is a Rust-native transliteration
of the core evaluator from <https://github.com/StanfordASL/stlcg>.

## Status

Implemented:

- predicates: `<=`, `>=`, equality robustness
- expression formulas, including direct identity-style signal formulas
- Boolean operators: negation, conjunction, disjunction, implication
- temporal operators: always, eventually, until, then
- intervals: unbounded, closed `[a, b]`, and `[a, infinity)`
- exact, distributed-exact, and smooth log-sum-exp aggregation
- 1D cumulative and finite-window integral formulas
- Burn autodiff compatibility
- pinned upstream golden fixtures generated from StanfordASL `stlcg.py`
- validation for malformed intervals and evaluation options

Not included in v0.1:

- graph visualization and notebook helpers
- translated application notebooks/examples
- the upstream experimental AGM mode

```rust
use burn::backend::Flex;
use burn::tensor::Tensor;
use stlcg::{EvalOptions, SignalEnv, Interval, var};

type B = Flex;

let device = Default::default();
let x = Tensor::<B, 3>::from_data([[[3.0_f32], [2.0], [1.0]]], &device);

let env = SignalEnv::new().with("x", x);
let formula = var("x").le(2.0).always(Interval::unbounded());
let trace = formula.robustness_trace(&env, EvalOptions::default()).unwrap();
```

Inputs are expected to be time-reversed, matching upstream STLCG semantics.
Unlike upstream, empty bounded `until`/`then` windows evaluate to negative
infinity rather than `-1e6` so very negative real robustness values are not
masked.

## API Overview

Create named signals with `SignalEnv`, build formulas with `var("name")`, and
evaluate robustness with `robustness_trace` or `robustness_at`.

```rust
use stlcg::{EvalOptions, Interval, SignalEnv, var};

let formula = var("x")
    .ge(0.0)
    .until(var("x").le(2.0), Interval::unbounded(), true);

let robustness = formula.robustness_trace(&env, EvalOptions::default())?;
```

Smooth robustness uses log-sum-exp:

```rust
let options = EvalOptions::smooth(10.0);
```

Negation is idiomatic Rust:

```rust
let formula = !var("x").le(2.0);
```

For direct identity-style formulas, convert an expression into a formula:

```rust
let integral = var("x").into_formula().integral(Default::default());
```

## Verification

Run the full local check suite:

```sh
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo rustdoc -- -D missing-docs
cargo package --no-verify
```

## Upstream Golden Fixtures

The fixture at `tests/fixtures/upstream_stlcg.json` is generated from a pinned
upstream StanfordASL Python implementation commit. To refresh it:

```sh
python3 -m venv /tmp/stlcg-upstream-venv
/tmp/stlcg-upstream-venv/bin/python -m pip install 'numpy<2' 'torch==2.2.2'
/tmp/stlcg-upstream-venv/bin/python tests/fixtures/generate_upstream.py
```

The generated fixture is committed so normal Rust test runs do not require
Python, PyTorch, or network access.

## License

MIT. The upstream StanfordASL STLCG MIT copyright notice is retained in
`LICENSE`.
