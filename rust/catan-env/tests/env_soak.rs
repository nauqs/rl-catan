//! Run-safety suite: long-haul soak with invariants on every finished
//! episode, batch determinism per seed, env outputs cross-checked against
//! the codec/encoder on the underlying game, and loud failure modes.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use catan_core::game::CatanGame;
use catan_env::{
    encode_obs, fill_action_mask, CatanEnv, EnvConfig, VecCatanEnv, NUM_ACTIONS, OBS_DIM,
};

fn pick_masked(rng: &mut SmallRng, mask: &[bool]) -> u32 {
    let legal = mask.iter().filter(|&&m| m).count();
    let k = rng.gen_range(0..legal);
    mask.iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .nth(k)
        .map(|(i, _)| i as u32)
        .unwrap()
}

/// The env's obs/mask must be exactly the encoder/codec applied to its own
/// game — no drift between the env layer and the primitives it wraps.
#[test]
fn env_outputs_match_codec_and_encoder_exactly() {
    let mut env = CatanEnv::new(EnvConfig::default(), 17);
    let mut rng = SmallRng::seed_from_u64(17);
    let (mut env_obs, mut raw_obs) = (vec![0.0f32; OBS_DIM], vec![0.0f32; OBS_DIM]);
    let mut env_mask = vec![false; NUM_ACTIONS];
    let mut raw_mask = [false; NUM_ACTIONS];
    let mut scratch = Vec::with_capacity(512);

    for step in 0..2_000 {
        env.write_obs(&mut env_obs);
        env.write_mask(&mut env_mask);

        let game: &CatanGame = env.game();
        encode_obs(
            game,
            game.current_player(),
            env.config.visibility,
            &mut raw_obs,
        );
        fill_action_mask(game, &mut scratch, &mut raw_mask);

        assert_eq!(
            env_obs, raw_obs,
            "step {step}: env obs drifted from encoder"
        );
        assert_eq!(
            &env_mask[..],
            &raw_mask[..],
            "step {step}: env mask drifted from codec"
        );

        let action = pick_masked(&mut rng, &env_mask);
        if env.step(action as usize).done {
            env.reset(rng.gen());
        }
    }
}

#[test]
fn vec_env_is_deterministic_per_seed() {
    const N: usize = 8;
    let run = |base_seed: u64| -> (Vec<f32>, Vec<u32>, u64) {
        let mut venv = VecCatanEnv::new(N, EnvConfig::default(), base_seed);
        let mut rng = SmallRng::seed_from_u64(123); // same policy stream both runs
        let mut obs = vec![0.0f32; N * OBS_DIM];
        let mut masks = vec![false; N * NUM_ACTIONS];
        let mut seats = vec![0u32; N];
        let mut rewards = vec![0.0f32; N];
        let mut dones = vec![false; N];
        let mut terminals = vec![0.0f32; N * 4];
        let mut actions = vec![0u32; N];

        venv.observe(&mut obs, &mut masks, &mut seats);
        let mut reward_trace = Vec::new();
        let mut seat_trace = Vec::new();
        let mut episodes = 0u64;
        for _ in 0..1_500 {
            for i in 0..N {
                actions[i] = pick_masked(&mut rng, &masks[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS]);
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
            reward_trace.extend_from_slice(&terminals);
            seat_trace.extend_from_slice(&seats);
            episodes += dones.iter().filter(|&&d| d).count() as u64;
        }
        (reward_trace, seat_trace, episodes)
    };

    let a = run(2026);
    let b = run(2026);
    assert_eq!(a.2, b.2, "episode counts diverged");
    assert_eq!(a.1, b.1, "seat sequences diverged");
    assert_eq!(a.0, b.0, "reward traces diverged");

    let c = run(2027);
    assert_ne!(a.1, c.1, "different seeds must produce different runs");
}

/// Long-haul soak: hundreds of auto-reset episodes; every finished episode
/// must satisfy the reward and termination invariants.
#[test]
fn soak_hundreds_of_episodes_uphold_invariants() {
    const N: usize = 32;
    let mut venv = VecCatanEnv::new(N, EnvConfig::default(), 99);
    let mut rng = SmallRng::seed_from_u64(99);
    let mut obs = vec![0.0f32; N * OBS_DIM];
    let mut masks = vec![false; N * NUM_ACTIONS];
    let mut seats = vec![0u32; N];
    let mut rewards = vec![0.0f32; N];
    let mut dones = vec![false; N];
    let mut terminals = vec![0.0f32; N * 4];
    let mut actions = vec![0u32; N];

    venv.observe(&mut obs, &mut masks, &mut seats);
    let (mut episodes, mut wins, mut truncations) = (0u32, 0u32, 0u32);

    while episodes < 250 {
        for i in 0..N {
            actions[i] = pick_masked(&mut rng, &masks[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS]);
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
        for i in 0..N {
            assert!(rewards[i].is_finite());
            if !dones[i] {
                continue;
            }
            episodes += 1;
            let row = &terminals[i * 4..i * 4 + 4];
            let sum: f32 = row.iter().sum();
            let max = row.iter().cloned().fold(f32::MIN, f32::max);
            if (sum + 2.0).abs() < 1e-4 && (max - 1.0).abs() < 1e-4 {
                wins += 1; // one winner (+1), three losers (-1)
            } else if row.iter().all(|&r| r == 0.0) {
                truncations += 1;
            } else {
                panic!("episode {episodes}: invalid terminal rewards {row:?}");
            }
            // Auto-reset must hand back a live decision point.
            let legal = masks[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS]
                .iter()
                .filter(|&&m| m)
                .count();
            assert!(legal >= 2, "post-reset mask not live");
        }
    }

    assert!(wins + truncations == episodes);
    assert!(
        (truncations as f32) < episodes as f32 * 0.2,
        "{truncations}/{episodes} truncations — random games should mostly finish"
    );
    println!("soak: {episodes} episodes, {wins} decisive, {truncations} truncated");
}

#[test]
#[should_panic(expected = "policy must sample under the legal mask")]
fn illegal_action_panics_loudly() {
    let mut env = CatanEnv::new(EnvConfig::default(), 1);
    let mut mask = vec![false; NUM_ACTIONS];
    env.write_mask(&mut mask);
    let illegal = (0..NUM_ACTIONS).find(|&i| !mask[i]).unwrap();
    env.step(illegal);
}

#[test]
#[should_panic(expected = "outside sane range")]
fn absurd_victory_target_panics_at_construction() {
    let config = EnvConfig {
        victory_target: 99,
        ..EnvConfig::default()
    };
    let _ = CatanEnv::new(config, 1);
}

#[test]
fn vec_env_replay_harvest_yields_replayable_episodes() {
    const N: usize = 8;
    let mut venv = VecCatanEnv::new(N, EnvConfig::default(), 31);
    venv.enable_recording();
    let mut rng = SmallRng::seed_from_u64(31);
    let mut obs = vec![0.0f32; N * OBS_DIM];
    let mut masks = vec![false; N * NUM_ACTIONS];
    let mut seats = vec![0u32; N];
    let mut rewards = vec![0.0f32; N];
    let mut dones = vec![false; N];
    let mut terminals = vec![0.0f32; N * 4];
    let mut actions = vec![0u32; N];

    venv.observe(&mut obs, &mut masks, &mut seats);
    let mut finished = 0;
    while finished < 10 {
        for i in 0..N {
            actions[i] = pick_masked(&mut rng, &masks[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS]);
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
        finished += dones.iter().filter(|&&d| d).count();
    }

    let records = venv.drain_records();
    assert!(
        records.len() >= 10,
        "harvested only {} records",
        records.len()
    );
    for (i, record) in records.iter().enumerate() {
        let bytes = record.to_bytes();
        let decoded = catan_core::replay::GameRecord::from_bytes(&bytes).unwrap();
        let replayed = decoded
            .replay()
            .unwrap_or_else(|e| panic!("record {i}: {e}"));
        assert_eq!(
            replayed.winner(),
            record.winner,
            "record {i}: winner mismatch"
        );
    }
    assert!(
        venv.drain_records().is_empty(),
        "drain must empty the harvest"
    );
}
