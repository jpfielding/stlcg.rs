use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData};
use serde::Deserialize;
use stlcg::{Aggregation, EvalOptions, Formula, IntegralOptions, Interval, SignalEnv, Trace, var};

type B = Flex;

#[derive(Debug, Deserialize)]
struct Fixture {
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
        let diff = (actual - expected).abs();
        assert!(
            diff <= 1.0e-4,
            "case `{name}` differed at flat index {index}: actual={actual}, expected={expected}, diff={diff}"
        );
    }
}
