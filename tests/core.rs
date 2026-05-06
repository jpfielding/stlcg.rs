use approx::assert_abs_diff_eq;
use burn::backend::{Autodiff, Flex};
use burn::tensor::Tensor;
use stlcg::{
    Aggregation, EvalOptions, IntegralOptions, IntegrationScheme, Interval, PaddingMode, SignalEnv,
    var,
};

type B = Flex;

fn env(values: [[[f32; 1]; 3]; 1]) -> SignalEnv<B> {
    let device = Default::default();
    SignalEnv::new().with("x", Tensor::<B, 3>::from_data(values, &device))
}

fn values(tensor: Tensor<B, 3>) -> Vec<f32> {
    tensor.into_data().into_vec::<f32>().unwrap()
}

fn assert_close(actual: Tensor<B, 3>, expected: &[f32]) {
    let actual = values(actual);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected.iter().copied()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-4);
    }
}

#[test]
fn predicate_and_temporal_exact_match_expected_reversed_trace_behavior() {
    let env = env([[[3.0], [2.0], [1.0]]]);

    assert_close(
        var("x")
            .le(2.0)
            .robustness_trace(&env, EvalOptions::default())
            .unwrap(),
        &[-1.0, 0.0, 1.0],
    );

    assert_close(
        var("x")
            .le(2.0)
            .always(Interval::unbounded())
            .robustness_trace(&env, EvalOptions::default())
            .unwrap(),
        &[-1.0, -1.0, -1.0],
    );

    assert_close(
        var("x")
            .le(2.0)
            .eventually(Interval::unbounded())
            .robustness_trace(&env, EvalOptions::default())
            .unwrap(),
        &[-1.0, 0.0, 1.0],
    );

    assert_close(
        var("x")
            .le(2.0)
            .always(Interval::closed(0, 1))
            .robustness_trace(&env, EvalOptions::default())
            .unwrap(),
        &[-1.0, -1.0, 0.0],
    );
}

#[test]
fn boolean_combinators_and_smooth_aggregation_work() {
    let env = env([[[3.0], [2.0], [1.0]]]);
    let formula = var("x").ge(2.0).and(var("x").le(2.0));
    assert_close(
        formula
            .robustness_trace(&env, EvalOptions::default())
            .unwrap(),
        &[-1.0, 0.0, -1.0],
    );

    let smooth = EvalOptions {
        predicate_scale: 1.0,
        aggregation: Aggregation::Smooth { scale: 10.0 },
    };
    let trace = var("x")
        .ge(2.0)
        .or(var("x").le(2.0))
        .robustness_trace(&env, smooth)
        .unwrap();
    let actual = values(trace);
    assert!(actual[0] > 0.99);
    assert!(actual[1] > 0.0);
    assert!(actual[2] > 0.99);
}

#[test]
fn integral_supports_cumulative_and_window_modes() {
    let env = env([[[1.0], [2.0], [3.0]]]);

    assert_close(
        var("x")
            .ge(0.0)
            .integral(IntegralOptions::cumulative())
            .robustness_trace(&env, EvalOptions::default())
            .unwrap(),
        &[1.0, 3.0, 6.0],
    );

    let opts = IntegralOptions {
        interval: Some(Interval::closed(0, 1)),
        use_relu: false,
        padding: PaddingMode::Same,
        scheme: IntegrationScheme::Riemann,
    };

    assert_close(
        var("x")
            .ge(0.0)
            .integral(opts)
            .robustness_trace(&env, EvalOptions::default())
            .unwrap(),
        &[2.0, 3.0, 5.0],
    );
}

#[test]
fn robustness_at_uses_non_reversed_time_indexing() {
    let env = env([[[3.0], [2.0], [1.0]]]);
    let at_now = var("x")
        .le(2.0)
        .robustness_at(&env, 0, EvalOptions::default())
        .unwrap();
    assert_close(at_now, &[1.0]);
}

#[test]
fn autodiff_backend_produces_gradients() {
    type AD = Autodiff<Flex>;

    let device = Default::default();
    let x = Tensor::<AD, 3>::from_data([[[1.0_f32], [2.0], [3.0]]], &device).require_grad();
    let env = SignalEnv::new().with("x", x.clone());
    let formula = var("x").ge(0.0).eventually(Interval::unbounded());
    let loss = formula
        .robustness_trace(&env, EvalOptions::smooth(5.0))
        .unwrap()
        .sum();

    let grads = loss.backward();
    let grad = x.grad(&grads).unwrap();
    let values = grad.into_data().into_vec::<f32>().unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
}
