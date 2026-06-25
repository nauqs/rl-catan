//! Bot seats: bot-controlled seats must play themselves at engine speed and
//! never surface to the policy; the bots must actually play their strategy;
//! mixed games stay deterministic and reward-balanced.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use catan_env::env::SeatKind;
use catan_env::{CatanEnv, EnvConfig, NUM_ACTIONS};

fn pick_masked(rng: &mut SmallRng, mask: &[bool]) -> usize {
    let legal = mask.iter().filter(|&&m| m).count();
    let k = rng.gen_range(0..legal);
    mask.iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .nth(k)
        .map(|(i, _)| i)
        .unwrap()
}

fn config_with(seats: [SeatKind; 4]) -> EnvConfig {
    EnvConfig {
        seat_kinds: seats,
        ..EnvConfig::default()
    }
}

/// Play episodes with the policy seat acting randomly; return per-seat wins.
fn run_games(seats: [SeatKind; 4], episodes: u32, seed: u64) -> ([u32; 4], u32) {
    let mut env = CatanEnv::new(config_with(seats), seed);
    let mut rng = SmallRng::seed_from_u64(seed * 3 + 1);
    let mut mask = vec![false; NUM_ACTIONS];
    let mut wins = [0u32; 4];
    let mut truncations = 0u32;
    let mut done_eps = 0u32;
    while done_eps < episodes {
        env.write_mask(&mut mask);
        // Every surfaced decision must belong to a policy seat.
        assert_eq!(
            seats[env.current_seat()],
            SeatKind::Policy,
            "bot seat {} surfaced to the policy",
            env.current_seat()
        );
        let action = pick_masked(&mut rng, &mask);
        let result = env.step(action);
        if result.done {
            done_eps += 1;
            if result.winner >= 0 {
                wins[result.winner as usize] += 1;
            } else {
                truncations += 1;
            }
            env.reset(seed + done_eps as u64 * 7919);
        }
    }
    (wins, truncations)
}

#[test]
fn bot_seats_never_surface_and_games_complete() {
    let (wins, truncations) = run_games(
        [
            SeatKind::Policy,
            SeatKind::HeuristicBot,
            SeatKind::RandomBot,
            SeatKind::HeuristicBot,
        ],
        20,
        5,
    );
    let total: u32 = wins.iter().sum();
    assert_eq!(total + truncations, 20);
}

#[test]
fn heuristic_bots_actually_play_heuristics() {
    // Random-acting policy seat + 1 random bot vs 2 heuristic bots: the
    // heuristic seats must dominate, proving the bots run their strategy
    // (not uniform fallback).
    let (wins, _) = run_games(
        [
            SeatKind::Policy, // random-acting in this test
            SeatKind::HeuristicBot,
            SeatKind::RandomBot,
            SeatKind::HeuristicBot,
        ],
        40,
        11,
    );
    let heuristic = wins[1] + wins[3];
    let randomish = wins[0] + wins[2];
    assert!(
        heuristic > randomish * 3,
        "heuristic bots should dominate random play: {wins:?}"
    );
}

#[test]
fn mixed_games_are_deterministic_per_seed() {
    let run = |seed: u64| {
        run_games(
            [
                SeatKind::Policy,
                SeatKind::HeuristicBot,
                SeatKind::HeuristicBot,
                SeatKind::HeuristicBot,
            ],
            10,
            seed,
        )
    };
    assert_eq!(run(42), run(42));
}

#[test]
#[should_panic(expected = "at least one seat must be policy-controlled")]
fn all_bot_configs_are_rejected() {
    let _ = CatanEnv::new(
        config_with([
            SeatKind::HeuristicBot,
            SeatKind::HeuristicBot,
            SeatKind::RandomBot,
            SeatKind::RandomBot,
        ]),
        1,
    );
}

#[test]
fn bot_games_still_balance_terminal_rewards() {
    let mut env = CatanEnv::new(
        config_with([
            SeatKind::Policy,
            SeatKind::HeuristicBot,
            SeatKind::HeuristicBot,
            SeatKind::HeuristicBot,
        ]),
        9,
    );
    let mut rng = SmallRng::seed_from_u64(9);
    let mut mask = vec![false; NUM_ACTIONS];
    let mut checked = 0;
    while checked < 5 {
        env.write_mask(&mut mask);
        let action = pick_masked(&mut rng, &mask);
        let result = env.step(action);
        if result.done {
            checked += 1;
            if result.winner >= 0 {
                let sum: f32 = result.terminal_rewards.iter().sum();
                assert!((sum + 2.0).abs() < 1e-4, "rewards {:?}", result.terminal_rewards);
            }
            env.reset(9 + checked as u64);
        }
    }
}
