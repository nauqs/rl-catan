//! Env-layer throughput: the full RL step (legality mask -> engine step ->
//! observation encode) and the batched vector path. These bound how fast
//! rollouts can feed the trainer.

use catan_env::{CatanEnv, EnvConfig, VecCatanEnv, NUM_ACTIONS, OBS_DIM};
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn first_legal(mask: &[bool]) -> usize {
    mask.iter().position(|&m| m).expect("live mask")
}

fn bench_single_env(c: &mut Criterion) {
    let mut env = CatanEnv::new(EnvConfig::default(), 1);
    let mut obs = vec![0.0f32; OBS_DIM];
    let mut mask = vec![false; NUM_ACTIONS];
    let mut episode = 0u64;
    c.bench_function("env/step_mask_obs", |b| {
        b.iter(|| {
            env.write_mask(&mut mask);
            let result = env.step(first_legal(&mask));
            if result.done {
                episode += 1;
                env.reset(episode);
            }
            env.write_obs(&mut obs);
            black_box(obs[0])
        })
    });
}

fn bench_vec_env(c: &mut Criterion) {
    const N: usize = 1024;
    let mut venv = VecCatanEnv::new(N, EnvConfig::default(), 9);
    let mut obs = vec![0.0f32; N * OBS_DIM];
    let mut masks = vec![false; N * NUM_ACTIONS];
    let mut seats = vec![0u32; N];
    let mut rewards = vec![0.0f32; N];
    let mut dones = vec![false; N];
    let mut terminals = vec![0.0f32; N * 4];
    let mut actions = vec![0u32; N];
    venv.observe(&mut obs, &mut masks, &mut seats);

    c.bench_function("env/vec1024_step_batch", |b| {
        b.iter(|| {
            for i in 0..N {
                actions[i] = first_legal(&masks[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS]) as u32;
            }
            venv.step_batch(
                &actions,
                &mut obs,
                &mut masks,
                &mut seats,
                &mut rewards,
                &mut dones,
                &mut terminals,
            );
            black_box(rewards[0])
        })
    });
}

criterion_group!(benches, bench_single_env, bench_vec_env);
criterion_main!(benches);
