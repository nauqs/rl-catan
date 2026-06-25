//! The RL hot loop (mask -> decode -> step -> auto-resolve -> obs encode)
//! must be allocation-free per decision in steady state, and the batched
//! path's per-batch allocations must stay bounded (lane setup + rayon
//! plumbing — never per-env). Own test binary: global allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use catan_env::{CatanEnv, EnvConfig, VecCatanEnv, NUM_ACTIONS, OBS_DIM};

struct CountingAllocator;
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

/// Allocation-free masked pick: count legal, choose k, second pass.
fn pick_masked(state: &mut u64, mask: &[bool]) -> usize {
    // xorshift, no rand dep needed here
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let legal = mask.iter().filter(|&&m| m).count();
    assert!(legal > 0);
    let k = (*state as usize) % legal;
    mask.iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .nth(k)
        .map(|(i, _)| i)
        .unwrap()
}

// NOTE: exactly ONE #[test] in this binary. The allocation counter is
// global; parallel test threads would bleed counts into each other's
// measurement windows.
#[test]
fn allocation_discipline() {
    single_env_decisions_allocate_nothing();
    vec_env_batch_allocations_stay_bounded();
}

fn single_env_decisions_allocate_nothing() {
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > 0,
        "allocator not engaged"
    );

    let mut env = CatanEnv::new(EnvConfig::default(), 99);
    let mut obs = vec![0.0f32; OBS_DIM];
    let mut mask = vec![false; NUM_ACTIONS];
    let mut rng_state = 0xDEADBEEFu64;

    // Warm up within one episode so all internal buffers reach capacity.
    let mut decision = |env: &mut CatanEnv,
                        mask: &mut Vec<bool>,
                        obs: &mut Vec<f32>,
                        rng_state: &mut u64|
     -> bool {
        env.write_mask(mask);
        let action = pick_masked(rng_state, mask);
        let result = env.step(action);
        env.write_obs(obs);
        result.done
    };

    let mut warmup = 0;
    while warmup < 300 && !decision(&mut env, &mut mask, &mut obs, &mut rng_state) {
        warmup += 1;
    }
    assert!(warmup >= 300, "episode too short for warmup");

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let mut measured = 0;
    while measured < 600 && !decision(&mut env, &mut mask, &mut obs, &mut rng_state) {
        measured += 1;
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
    assert!(
        measured >= 200,
        "episode ended too early ({measured} decisions)"
    );
    assert_eq!(
        allocations, 0,
        "{allocations} heap allocations across {measured} steady-state decisions"
    );
}

fn vec_env_batch_allocations_stay_bounded() {
    const N: usize = 32;
    let mut venv = VecCatanEnv::new(N, EnvConfig::default(), 7);
    let mut obs = vec![0.0f32; N * OBS_DIM];
    let mut masks = vec![false; N * NUM_ACTIONS];
    let mut seats = vec![0u32; N];
    let mut rewards = vec![0.0f32; N];
    let mut dones = vec![false; N];
    let mut terminals = vec![0.0f32; N * 4];
    let mut actions = vec![0u32; N];
    let mut rng_state = 0xC0FFEEu64;

    venv.observe(&mut obs, &mut masks, &mut seats);
    let mut run_batch = |venv: &mut VecCatanEnv| {
        for i in 0..N {
            actions[i] = pick_masked(
                &mut rng_state,
                &masks[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS],
            ) as u32;
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
        dones.iter().filter(|&&d| d).count()
    };

    // Warm up rayon's pools and all env buffers.
    for _ in 0..50 {
        run_batch(&mut venv);
    }

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let mut batches = 0u64;
    let mut resets = 0usize;
    for _ in 0..100 {
        resets += run_batch(&mut venv);
        batches += 1;
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;

    // Budget: lane Vec + rayon job plumbing per batch, plus episode resets
    // (game construction allocates a few). NEVER proportional to N per env
    // step — that's the regression this guards against.
    let budget = batches * 24 + resets as u64 * 16;
    assert!(
        allocations <= budget,
        "{allocations} allocations over {batches} batches ({resets} resets) — \
         budget {budget}; a per-env-step allocation crept in"
    );
}
