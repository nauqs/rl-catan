//! Env semantics: masked random policies must complete episodes; forced
//! moves never reach the policy; reward bookkeeping must balance exactly;
//! the vector env must auto-reset and stay deterministic per seed.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use catan_env::{CatanEnv, EnvConfig, RewardConfig, VecCatanEnv, NUM_ACTIONS, OBS_DIM};

fn pick_masked(rng: &mut SmallRng, mask: &[bool]) -> usize {
    let legal: Vec<usize> = (0..NUM_ACTIONS).filter(|&i| mask[i]).collect();
    assert!(
        !legal.is_empty(),
        "no legal actions on a live decision point"
    );
    legal[rng.gen_range(0..legal.len())]
}

/// Play one full episode with a masked random policy; returns
/// (winner, per-seat delivered reward totals, decision count, final VPs).
fn run_episode(env: &mut CatanEnv, policy_seed: u64) -> (i8, [f32; 4], u64, [i32; 4]) {
    let mut rng = SmallRng::seed_from_u64(policy_seed);
    let mut mask = vec![false; NUM_ACTIONS];
    let mut delivered = [0.0f32; 4];
    let mut decisions = 0u64;

    let mut seat = env.current_seat();
    loop {
        env.write_mask(&mut mask);
        let action = pick_masked(&mut rng, &mask);
        let result = env.step(action);
        decisions += 1;
        assert!(decisions < 50_000, "episode failed to terminate");
        if result.done {
            for p in 0..4 {
                delivered[p] += result.terminal_rewards[p];
            }
            let mut vps = [0i32; 4];
            for (p, vp) in vps.iter_mut().enumerate().take(env.config.num_players) {
                *vp = env.game().state.calculate_victory_points(p);
            }
            return (result.winner, delivered, decisions, vps);
        }
        delivered[result.seat] += result.reward;
        seat = result.seat;
        let _ = seat;
    }
}

#[test]
fn masked_random_policy_completes_episodes_with_balanced_rewards() {
    for seed in [1u64, 5, 9] {
        let mut env = CatanEnv::new(EnvConfig::default(), seed);
        let (winner, delivered, _, _) = run_episode(&mut env, seed * 7);
        if winner >= 0 {
            // Terminal-only rewards: winner +1, the other three -1 each.
            assert_eq!(
                delivered[winner as usize], 1.0,
                "seed {seed}: winner reward"
            );
            let total: f32 = delivered.iter().sum();
            assert!(
                (total - (1.0 - 3.0)).abs() < 1e-5,
                "seed {seed}: rewards must sum to -2"
            );
        } else {
            assert!(
                delivered.iter().all(|&r| r == 0.0),
                "truncation pays nothing"
            );
        }
    }
}

#[test]
fn every_policy_decision_has_at_least_two_choices() {
    // The whole point of auto_resolve_forced: non-decisions never reach the
    // policy.
    let mut env = CatanEnv::new(EnvConfig::default(), 3);
    let mut rng = SmallRng::seed_from_u64(11);
    let mut mask = vec![false; NUM_ACTIONS];
    for _ in 0..5_000 {
        env.write_mask(&mut mask);
        let legal = mask.iter().filter(|&&m| m).count();
        assert!(
            legal >= 2,
            "forced move leaked to the policy ({legal} legal)"
        );
        let action = pick_masked(&mut rng, &mask);
        if env.step(action).done {
            env.reset(rng.gen());
        }
    }
}

#[test]
fn shaped_rewards_account_exactly_for_final_vp() {
    // With vp_delta = 0.1 and no terminal reward, each seat's delivered
    // total must equal 0.1 * its final VP (every point is accrued once and
    // flushed by terminal_rewards at the end).
    let config = EnvConfig {
        reward: RewardConfig {
            win: 0.0,
            loss: 0.0,
            vp_delta: 0.1,
        },
        ..EnvConfig::default()
    };
    for seed in [2u64, 8] {
        let mut env = CatanEnv::new(config, seed);
        let (_, delivered, _, vps) = run_episode(&mut env, seed);
        for p in 0..4 {
            let expected = 0.1 * vps[p] as f32;
            assert!(
                (delivered[p] - expected).abs() < 1e-4,
                "seed {seed} seat {p}: delivered {} != 0.1 * {} VP",
                delivered[p],
                vps[p]
            );
        }
    }
}

#[test]
fn truncation_ends_episodes_without_winner() {
    let config = EnvConfig {
        max_turns: 15,
        ..EnvConfig::default()
    };
    let mut env = CatanEnv::new(config, 4);
    let (winner, delivered, decisions, _) = run_episode(&mut env, 4);
    assert_eq!(winner, -1, "15-turn cap cannot produce a winner");
    assert!(delivered.iter().all(|&r| r == 0.0));
    assert!(decisions < 2_000, "truncated episode should be short");
}

#[test]
fn same_seed_same_episode() {
    let run = |env_seed: u64| -> (i8, u64, [i32; 4]) {
        let mut env = CatanEnv::new(EnvConfig::default(), env_seed);
        let (winner, _, decisions, vps) = run_episode(&mut env, 99);
        (winner, decisions, vps)
    };
    assert_eq!(
        run(42),
        run(42),
        "identical seeds + policy must reproduce the episode"
    );
}

#[test]
fn first_to_seven_config_flows_through() {
    let config = EnvConfig {
        victory_target: 7,
        ..EnvConfig::default()
    };
    let mut env = CatanEnv::new(config, 6);
    let (winner, _, _, vps) = run_episode(&mut env, 6);
    if winner >= 0 {
        assert!(
            (7..10).contains(&vps[winner as usize]),
            "first-to-7 winner at {} VP",
            vps[winner as usize]
        );
    }
}

#[test]
fn recording_captures_replayable_episodes() {
    let mut env = CatanEnv::new(EnvConfig::default(), 13);
    env.enable_recording();
    env.reset(13);
    let (winner, ..) = run_episode(&mut env, 13);
    let record = env.take_record().expect("recording was enabled");
    assert_eq!(record.winner, winner);
    let replayed = record.replay().expect("env episode must replay exactly");
    assert_eq!(replayed.winner(), winner);
    assert_eq!(replayed.state.vertices, env.game().state.vertices);
}

#[test]
fn vec_env_steps_batches_and_auto_resets() {
    const N: usize = 16;
    let mut venv = VecCatanEnv::new(N, EnvConfig::default(), 77);
    let mut obs = vec![0.0f32; N * OBS_DIM];
    let mut masks = vec![false; N * NUM_ACTIONS];
    let mut seats = vec![0u32; N];
    let mut rewards = vec![0.0f32; N];
    let mut dones = vec![false; N];
    let mut terminals = vec![0.0f32; N * 4];
    let mut actions = vec![0u32; N];
    let mut rng = SmallRng::seed_from_u64(5);

    venv.observe(&mut obs, &mut masks, &mut seats);

    let mut finished_episodes = 0u32;
    let mut terminal_sum = 0.0f32;
    for step in 0..30_000 {
        for i in 0..N {
            actions[i] =
                pick_masked(&mut rng, &masks[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS]) as u32;
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
            assert!((seats[i] as usize) < 4);
            if dones[i] {
                finished_episodes += 1;
                terminal_sum += terminals[i * 4..i * 4 + 4].iter().sum::<f32>();
                // Auto-reset: the obs/mask already belong to a fresh episode.
                let legal = masks[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS]
                    .iter()
                    .filter(|&&m| m)
                    .count();
                assert!(legal >= 2, "post-reset mask must be live");
            }
        }
        if finished_episodes >= 12 && step > 100 {
            break;
        }
    }
    assert!(
        finished_episodes >= 12,
        "only {finished_episodes} episodes finished"
    );
    // Every finished 4p game with a winner sums to 1 - 3 = -2; truncations 0.
    assert!(
        terminal_sum <= -2.0 * (finished_episodes as f32 - 2.0),
        "terminal reward sums look wrong: {terminal_sum} over {finished_episodes} episodes"
    );
}
