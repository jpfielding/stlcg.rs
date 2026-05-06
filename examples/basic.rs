use burn::backend::Flex;
use burn::tensor::Tensor;
use stlcg::{EvalOptions, Interval, SignalEnv, var};

type B = Flex;

fn main() {
    let device = Default::default();
    let x = Tensor::<B, 3>::from_data([[[3.0_f32], [2.0], [1.0]]], &device);

    let env = SignalEnv::new().with("x", x);
    let formula = var("x").le(2.0).always(Interval::unbounded());
    let trace = formula
        .robustness_trace(&env, EvalOptions::default())
        .unwrap();

    println!("{trace}");
}
