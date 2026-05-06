use burn::tensor::{Bool, Tensor, activation, backend::Backend};

use crate::{Aggregation, Interval, Trace};

pub(crate) fn maxish<B: Backend>(
    x: Tensor<B, 4>,
    dim: usize,
    aggregation: Aggregation,
) -> Tensor<B, 4> {
    match aggregation {
        Aggregation::Exact => x.max_dim(dim),
        Aggregation::DistributedExact => distributed_extreme(x, dim, true),
        Aggregation::Smooth { scale } => logsumexp(x * scale, dim) / scale,
    }
}

pub(crate) fn minish<B: Backend>(
    x: Tensor<B, 4>,
    dim: usize,
    aggregation: Aggregation,
) -> Tensor<B, 4> {
    match aggregation {
        Aggregation::Exact => x.min_dim(dim),
        Aggregation::DistributedExact => distributed_extreme(x, dim, false),
        Aggregation::Smooth { scale } => logsumexp(x.neg() * scale, dim).neg() / scale,
    }
}

pub(crate) fn maxish3<B: Backend>(x: Trace<B>, dim: usize, aggregation: Aggregation) -> Trace<B> {
    match aggregation {
        Aggregation::Exact => x.max_dim(dim),
        Aggregation::DistributedExact => distributed_extreme3(x, dim, true),
        Aggregation::Smooth { scale } => logsumexp3(x * scale, dim) / scale,
    }
}

pub(crate) fn minish3<B: Backend>(x: Trace<B>, dim: usize, aggregation: Aggregation) -> Trace<B> {
    match aggregation {
        Aggregation::Exact => x.min_dim(dim),
        Aggregation::DistributedExact => distributed_extreme3(x, dim, false),
        Aggregation::Smooth { scale } => logsumexp3(x.neg() * scale, dim).neg() / scale,
    }
}

fn logsumexp<B: Backend>(x: Tensor<B, 4>, dim: usize) -> Tensor<B, 4> {
    let max = x.clone().max_dim(dim);
    ((x - max.clone()).exp().sum_dim(dim).log()) + max
}

fn logsumexp3<B: Backend>(x: Trace<B>, dim: usize) -> Trace<B> {
    let max = x.clone().max_dim(dim);
    ((x - max.clone()).exp().sum_dim(dim).log()) + max
}

fn distributed_extreme<B: Backend>(x: Tensor<B, 4>, dim: usize, is_max: bool) -> Tensor<B, 4> {
    let extreme = if is_max {
        x.clone().max_dim(dim)
    } else {
        x.clone().min_dim(dim)
    };
    let mask = x.clone().equal(extreme);
    masked_mean(x, mask, dim)
}

fn distributed_extreme3<B: Backend>(x: Trace<B>, dim: usize, is_max: bool) -> Trace<B> {
    let extreme = if is_max {
        x.clone().max_dim(dim)
    } else {
        x.clone().min_dim(dim)
    };
    let mask = x.clone().equal(extreme);
    masked_mean3(x, mask, dim)
}

fn masked_mean<B: Backend>(x: Tensor<B, 4>, mask: Tensor<B, 4, Bool>, dim: usize) -> Tensor<B, 4> {
    let weights = mask.float();
    let sum = (x * weights.clone()).sum_dim(dim);
    let count = weights.sum_dim(dim);
    sum / count
}

fn masked_mean3<B: Backend>(x: Trace<B>, mask: Tensor<B, 3, Bool>, dim: usize) -> Trace<B> {
    let weights = mask.float();
    let sum = (x * weights.clone()).sum_dim(dim);
    let count = weights.sum_dim(dim);
    sum / count
}

pub(crate) fn stack_last<B: Backend>(tensors: Vec<Trace<B>>) -> Tensor<B, 4> {
    let tensors = tensors
        .into_iter()
        .map(|tensor| tensor.unsqueeze_dim::<4>(3))
        .collect();
    Tensor::cat(tensors, 3)
}

pub(crate) fn reduce_last_min<B: Backend>(
    tensors: Vec<Trace<B>>,
    aggregation: Aggregation,
) -> Trace<B> {
    minish(stack_last(tensors), 3, aggregation).squeeze_dim::<3>(3)
}

pub(crate) fn reduce_last_max<B: Backend>(
    tensors: Vec<Trace<B>>,
    aggregation: Aggregation,
) -> Trace<B> {
    maxish(stack_last(tensors), 3, aggregation).squeeze_dim::<3>(3)
}

pub(crate) fn select_time<B: Backend>(trace: &Trace<B>, index: usize) -> Trace<B> {
    trace.clone().narrow(1, index, 1)
}

pub(crate) fn window_for_interval<B: Backend>(
    trace: &Trace<B>,
    index: usize,
    interval: Interval,
) -> Trace<B> {
    match interval {
        Interval::Unbounded => trace.clone().narrow(1, 0, index + 1),
        Interval::Closed { start, end } => {
            let slices = (start..=end)
                .map(|offset| select_time(trace, index.saturating_sub(offset)))
                .collect();
            Tensor::cat(slices, 1)
        }
        Interval::From { start } => trace.clone().narrow(1, 0, index.saturating_sub(start) + 1),
    }
}

pub(crate) fn temporal<B: Backend>(
    trace: Trace<B>,
    interval: Interval,
    aggregation: Aggregation,
    is_always: bool,
) -> Trace<B> {
    let time = trace.dims()[1];
    let outputs = (0..time)
        .map(|i| {
            let window = window_for_interval(&trace, i, interval);
            if is_always {
                minish3(window, 1, aggregation)
            } else {
                maxish3(window, 1, aggregation)
            }
        })
        .collect();

    Tensor::cat(outputs, 1)
}

pub(crate) fn relu<B: Backend>(trace: Trace<B>) -> Trace<B> {
    activation::relu(trace)
}
