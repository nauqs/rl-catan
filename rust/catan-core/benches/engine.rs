//! Criterion micro-benchmarks for the engine hot path.
//!
//! Workflow:
//!   cargo bench -p catan-core -- --save-baseline before   # before a change
//!   cargo bench -p catan-core -- --baseline before        # compare after
//!
//! Benches cover the profiler-identified hot spots: main-phase legal-mask
//! generation (the dominant engine cost), action execution including the
//! longest-road path, and a dense worst-case longest-road recompute that
//! bounds MCTS tail latency.

use catan_core::board::topology;
use catan_core::building::{build_road, build_settlement, longest_road_length};
use catan_core::game::{Action, CatanGame, GamePhase, TurnPhase};
use catan_core::players::{HeuristicPlayer, Player, RandomPlayer};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::hint::black_box;

/// Advance a game with the given players until (at least) `min_steps` have
/// run AND the game sits in the main phase. None if the game ends first.
fn try_midgame_at_main(seed: u64, min_steps: usize, heuristic: bool) -> Option<CatanGame> {
    let mut game = CatanGame::new(4, seed);
    game.record_history = false;
    let mut players: Vec<Box<dyn Player>> = (0..4u64)
        .map(|i| -> Box<dyn Player> {
            if heuristic {
                Box::new(HeuristicPlayer::new(seed * 17 + i))
            } else {
                Box::new(RandomPlayer::new(seed * 17 + i))
            }
        })
        .collect();

    let mut valid = Vec::with_capacity(256);
    let mut steps = 0usize;
    loop {
        let settled = steps >= min_steps
            && game.game_phase == GamePhase::Playing
            && game.turn_phase == TurnPhase::Main;
        if settled {
            return Some(game);
        }
        if game.is_game_over() {
            return None;
        }
        game.fill_valid_actions(&mut valid);
        if valid.is_empty() {
            return None;
        }
        let idx = game.current_player();
        let action = players[idx].choose_action(&game, &valid);
        game.execute_action(&action);
        steps += 1;
    }
}

/// First seed whose game reaches a main-phase state after `min_steps`.
fn midgame_at_main(min_steps: usize, heuristic: bool) -> CatanGame {
    (1..200u64)
        .find_map(|seed| try_midgame_at_main(seed, min_steps, heuristic))
        .expect("no seed produced a long enough game")
}

/// Same state but the player to act can afford everything: maximal mask
/// (all builds, bank trades, proposals). The realistic worst case per step.
fn rich(mut game: CatanGame) -> CatanGame {
    let p = game.state.current_player;
    game.state.resources[p] = [4, 4, 4, 4, 4];
    game
}

/// A state with a long road snake for one player: worst case for the
/// longest-road DFS.
fn dense_road_state() -> catan_core::state::GameState {
    let mut state = catan_core::state::GameState::new(4, 5);
    let _topo = topology();
    let start = 27u8; // interior vertex: room to snake in every direction
    assert!(build_settlement(&mut state, 0, start as usize, true));

    // Backtracking search for a 15-edge trail (no repeated edges).
    fn find_trail(at: u8, trail: &mut Vec<u8>, target: usize) -> bool {
        if trail.len() == target {
            return true;
        }
        let topo = topology();
        for &e in &topo.vertex_edges[at as usize] {
            if e >= 0 && !trail.contains(&(e as u8)) {
                let [a, b] = topo.edge_vertices[e as usize];
                let far = if a == at { b } else { a };
                trail.push(e as u8);
                if find_trail(far, trail, target) {
                    return true;
                }
                trail.pop();
            }
        }
        false
    }

    let mut trail = Vec::new();
    assert!(find_trail(start, &mut trail, 15), "no 15-edge trail found");
    for &e in &trail {
        assert!(build_road(&mut state, 0, e as usize, true));
    }
    state
}

fn bench_mask(c: &mut Criterion) {
    let midgame = midgame_at_main(600, false);
    let dense_rich = rich(midgame_at_main(250, true));
    let midgame_rich = rich(midgame.clone());
    let mut buf = Vec::with_capacity(512);

    c.bench_function("mask/main_midgame_typical", |b| {
        b.iter(|| {
            midgame.fill_valid_actions(&mut buf);
            black_box(buf.len())
        })
    });
    c.bench_function("mask/main_midgame_rich", |b| {
        b.iter(|| {
            midgame_rich.fill_valid_actions(&mut buf);
            black_box(buf.len())
        })
    });
    c.bench_function("mask/main_dense_board_rich", |b| {
        b.iter(|| {
            dense_rich.fill_valid_actions(&mut buf);
            black_box(buf.len())
        })
    });
}

fn bench_execute(c: &mut Criterion) {
    // Roll + resource distribution (the most common action).
    let mut roll_game = midgame_at_main(500, false);
    let valid = roll_game.valid_actions();
    let end = valid
        .iter()
        .find(|a| matches!(a, Action::EndTurn { .. }))
        .copied()
        .unwrap();
    roll_game.execute_action(&end); // now in PreRoll for the next player
    let roller = roll_game.current_player() as u8;
    c.bench_function("execute/roll_and_distribute", |b| {
        b.iter_batched(
            || roll_game.clone(),
            |mut g| {
                g.execute_action(&Action::RollDice {
                    player: roller,
                    forced: Some(6),
                });
                black_box(g.state.bank[0])
            },
            BatchSize::SmallInput,
        )
    });

    // Build a road when the builder already has a long network (triggers the
    // longest-road update path).
    let road_game = rich(midgame_at_main(250, true));
    let road = road_game
        .valid_actions()
        .iter()
        .find(|a| matches!(a, Action::BuildRoad { .. }))
        .copied();
    if let Some(road) = road {
        c.bench_function("execute/build_road_with_award_update", |b| {
            b.iter_batched(
                || road_game.clone(),
                |mut g| {
                    assert!(g.execute_action(&road));
                    black_box(g.state.longest_road_length)
                },
                BatchSize::SmallInput,
            )
        });
    }
}

fn bench_longest_road(c: &mut Criterion) {
    let state = dense_road_state();
    c.bench_function("longest_road/15_road_snake", |b| {
        b.iter(|| black_box(longest_road_length(&state, 0)))
    });
}

fn bench_full_game(c: &mut Criterion) {
    c.bench_function("game/full_random_4p", |b| {
        b.iter(|| {
            let mut game = CatanGame::new(4, 42);
            game.record_history = false;
            let mut players: Vec<Box<dyn Player>> = (0..4u64)
                .map(|i| Box::new(RandomPlayer::new(100 + i)) as Box<dyn Player>)
                .collect();
            let mut valid = Vec::with_capacity(256);
            while !game.is_game_over() && game.state.turn < 1000 {
                game.fill_valid_actions(&mut valid);
                if valid.is_empty() {
                    break;
                }
                let idx = game.current_player();
                let action = players[idx].choose_action(&game, &valid);
                game.execute_action(&action);
            }
            black_box(game.winner())
        })
    });
}

criterion_group!(
    benches,
    bench_mask,
    bench_execute,
    bench_longest_road,
    bench_full_game
);
criterion_main!(benches);
