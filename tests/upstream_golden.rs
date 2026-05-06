use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData};
use serde::Deserialize;
use stlcg::{Aggregation, EvalOptions, Formula, IntegralOptions, Interval, SignalEnv, Trace, var};

type B = Flex;
const SOURCE_COMMIT: &str = "abd16c92108f1b57a72d66c58492c949b6c5a8ea";

#[derive(Debug, Deserialize)]
struct Fixture {
    source_commit: String,
    source_url: String,
    trace: TensorFixture,
    cases: Vec<CaseFixture>,
}

#[derive(Debug, Deserialize)]
struct CaseFixture {
    name: String,
    expected: TensorFixture,
}

#[derive(Debug, Deserialize)]
struct TensorFixture {
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[test]
fn rust_matches_upstream_stlcg_golden_cases() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/upstream_stlcg.json")).unwrap();
    assert_eq!(fixture.trace.shape, [1, 4, 1]);
    assert_eq!(fixture.source_commit, SOURCE_COMMIT);
    assert!(fixture.source_url.contains(SOURCE_COMMIT));

    let device = Default::default();
    let trace = Tensor::<B, 3>::from_data(
        TensorData::new(fixture.trace.values.clone(), [1, 4, 1]),
        &device,
    );
    let env = SignalEnv::new().with("x", trace);

    for case in fixture.cases {
        let (formula, options) = case_formula(&case.name);
        let actual = formula.robustness_trace(&env, options).unwrap();
        assert_tensor_close(&case.name, actual, &case.expected);
    }
}

fn case_formula(name: &str) -> (Formula, EvalOptions) {
    let exact = EvalOptions::default();
    match name {
        "less_equal_x_le_2" => (var("x").le(2.0), exact),
        "greater_equal_x_ge_2" => (var("x").ge(2.0), exact),
        "equal_x_eq_2" => (var("x").eq_value(2.0), exact),
        "not_less_equal_x_le_2" => (!var("x").le(2.0), exact),
        "and_ge2_le2" => (var("x").ge(2.0).and(var("x").le(2.0)), exact),
        "or_ge2_le2" => (var("x").ge(2.0).or(var("x").le(2.0)), exact),
        "implies_ge2_le2" => (var("x").ge(2.0).implies(var("x").le(2.0)), exact),
        "always_unbounded_le2" => (var("x").le(2.0).always(Interval::unbounded()), exact),
        "eventually_unbounded_le2" => (var("x").le(2.0).eventually(Interval::unbounded()), exact),
        "always_closed_0_1_le2" => (var("x").le(2.0).always(Interval::closed(0, 1)), exact),
        "eventually_closed_0_1_le2" => (var("x").le(2.0).eventually(Interval::closed(0, 1)), exact),
        "always_from_1_le2" => (var("x").le(2.0).always(Interval::from(1)), exact),
        "eventually_from_1_le2" => (var("x").le(2.0).eventually(Interval::from(1)), exact),
        "or_smooth_scale_10" => (
            var("x").ge(2.0).or(var("x").le(2.0)),
            EvalOptions {
                predicate_scale: 1.0,
                aggregation: Aggregation::Smooth { scale: 10.0 },
            },
        ),
        "until_unbounded_ge2_until_le2" => (
            var("x")
                .ge(2.0)
                .until(var("x").le(2.0), Interval::unbounded(), true),
            exact,
        ),
        "then_unbounded_ge2_then_le2" => (
            var("x")
                .ge(2.0)
                .then(var("x").le(2.0), Interval::unbounded(), true),
            exact,
        ),
        "until_closed_1_2_ge2_until_le2" => (
            var("x")
                .ge(2.0)
                .until(var("x").le(2.0), Interval::closed(1, 2), true),
            exact,
        ),
        "then_closed_1_2_ge2_then_le2" => (
            var("x")
                .ge(2.0)
                .then(var("x").le(2.0), Interval::closed(1, 2), true),
            exact,
        ),
        "until_from_1_ge2_until_le2" => (
            var("x")
                .ge(2.0)
                .until(var("x").le(2.0), Interval::from(1), true),
            exact,
        ),
        "then_from_1_ge2_then_le2" => (
            var("x")
                .ge(2.0)
                .then(var("x").le(2.0), Interval::from(1), true),
            exact,
        ),
        "until_unbounded_no_overlap_ge2_until_le2" => (
            var("x")
                .ge(2.0)
                .until(var("x").le(2.0), Interval::unbounded(), false),
            exact,
        ),
        "then_unbounded_no_overlap_ge2_then_le2" => (
            var("x")
                .ge(2.0)
                .then(var("x").le(2.0), Interval::unbounded(), false),
            exact,
        ),
        "integral_identity_cumulative" => (
            var("x")
                .into_formula()
                .integral(IntegralOptions::cumulative()),
            exact,
        ),
        "integral_identity_window_0_1" => (
            var("x")
                .into_formula()
                .integral(IntegralOptions::window(Interval::closed(0, 1))),
            exact,
        ),
        unknown => panic!("unknown upstream fixture case `{unknown}`"),
    }
}

fn assert_tensor_close(name: &str, actual: Trace<B>, expected: &TensorFixture) {
    assert_eq!(
        actual.dims().as_slice(),
        expected.shape.as_slice(),
        "{name}"
    );
    let actual = actual.into_data().into_vec::<f32>().unwrap();
    assert_eq!(actual.len(), expected.values.len(), "{name}");
    for (index, (actual, expected)) in actual.iter().zip(expected.values.iter()).enumerate() {
        if is_empty_temporal_window_sentinel(*actual, *expected) {
            continue;
        }

        let diff = (actual - expected).abs();
        assert!(
            diff <= 1.0e-4,
            "case `{name}` differed at flat index {index}: actual={actual}, expected={expected}, diff={diff}"
        );
    }
}

fn is_empty_temporal_window_sentinel(actual: f32, expected: f32) -> bool {
    // Upstream uses -1e6 for empty bounded until/then windows; this crate uses
    // negative infinity so very negative real robustness values cannot be masked.
    expected == -1_000_000.0 && actual.is_infinite() && actual.is_sign_negative()
}
