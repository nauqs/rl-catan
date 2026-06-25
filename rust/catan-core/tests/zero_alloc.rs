//! Performance-as-a-test: in steady state the engine must take ZERO heap
//! allocations per step (mask generation + agent choice + execution).
//! The hot loop was engineered allocation-free; this test stops a future
//! change from quietly reintroducing a per-step Vec.
//!
//! Implementation: a counting global allocator (per test binary, which is
//! why this file holds only these tests).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use catan_core::game::{Action, CatanGame, GamePhase, TurnPhase};
use catan_core::players::{HeuristicPlayer, Player, RandomPlayer};

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

fn step(game: &mut CatanGame, players: &mut [Box<dyn Player>], valid: &mut Vec<Action>) -> bool {
    if game.is_game_over() || game.state.turn >= 1000 {
        return false;
    }
    let idx = game.current_player();
    game.fill_valid_actions(valid);
    if valid.is_empty() {
        return false;
    }
    let action = players[idx].choose_action(game, valid);
    game.execute_action(&action);
    true
}

/// Play `seed`'s game to the END, allowing allocations only during the first
/// `warmup_steps` (buffers growing to steady-state capacity). Everything
/// after — including every 7-roll discard cascade, robber move, steal, and
/// trade negotiation the game contains — must allocate nothing.
fn run_full_game(seed: u64, heuristic: bool, warmup_steps: usize) -> (u64, usize) {
    let mut game = CatanGame::new(4, seed);
    game.record_history = false;
    let mut players: Vec<Box<dyn Player>> = (0..4u64)
        .map(|i| -> Box<dyn Player> {
            if heuristic {
                Box::new(HeuristicPlayer::new(seed * 7 + i))
            } else {
                Box::new(RandomPlayer::new(seed * 7 + i))
            }
        })
        .collect();
    let mut valid = Vec::with_capacity(512);

    // Sanity: the counting allocator must be engaged (setup allocated).
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > 0,
        "counting allocator not engaged — the test would be vacuous"
    );

    let mut warmup = 0;
    while warmup < warmup_steps && step(&mut game, &mut players, &mut valid) {
        warmup += 1;
    }

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let mut measured = 0usize;
    while step(&mut game, &mut players, &mut valid) {
        measured += 1;
    }
    (ALLOCATIONS.load(Ordering::Relaxed) - before, measured)
}

// NOTE: exactly ONE #[test] in this binary — the global allocation counter
// must not be shared across parallel test threads.
#[test]
fn allocation_discipline() {
    full_games_allocate_nothing_after_warmup();
    robber_cycle_allocates_nothing_after_first();
}

fn full_games_allocate_nothing_after_warmup() {
    let mut total_steps = 0usize;
    // Random players hit trades, discards, robber, and steals constantly.
    for seed in [99u64, 7, 21, 42] {
        let (allocations, measured) = run_full_game(seed, false, 200);
        println!("seed {seed} (random): {allocations} allocations in {measured} measured steps");
        assert_eq!(
            allocations, 0,
            "seed {seed}: {allocations} heap allocations in {measured} steady-state steps"
        );
        total_steps += measured;
    }
    // Heuristic players exercise the scoring paths.
    for seed in [3u64, 11] {
        let (allocations, measured) = run_full_game(seed, true, 150);
        println!("seed {seed} (heuristic): {allocations} allocations in {measured} measured steps");
        assert_eq!(allocations, 0, "seed {seed}: heuristic game allocated");
        total_steps += measured;
    }
    assert!(
        total_steps > 8_000,
        "only {total_steps} steps measured — sample too small to trust"
    );
}

/// Deterministically drive the historically-allocating paths: a 7-roll with
/// multi-player discards, robber movement, and a steal. After one warmup
/// cycle (buffers sized), a second full cycle must allocate nothing.
fn robber_cycle_allocates_nothing_after_first() {
    let mut game = CatanGame::new(4, 13);
    game.record_history = false;
    game.game_phase = GamePhase::Playing;
    game.state.phase = 1;
    game.state.current_player = 0;

    // Buildings so steals have victims (spaced vertices on the same tile).
    use catan_core::board::topology;
    use catan_core::building::build_settlement;
    let verts = topology().tile_vertices[9];
    assert!(build_settlement(
        &mut game.state,
        1,
        verts[0] as usize,
        true
    ));
    assert!(build_settlement(
        &mut game.state,
        2,
        verts[2] as usize,
        true
    ));

    let mut valid = Vec::with_capacity(64);
    let cycle = |game: &mut CatanGame, valid: &mut Vec<Action>| {
        // Hands over the limit so both players must discard.
        for p in [1usize, 2] {
            let need = 9 - game.state.total_resources(p);
            for _ in 0..need {
                game.state.resources[p][0] += 1;
                game.state.bank[0] -= 1;
            }
        }
        game.turn_phase = TurnPhase::MustRoll;
        assert!(game.execute_action(&Action::RollDice {
            player: 0,
            forced: Some(7)
        }));
        while game.turn_phase == TurnPhase::RobberDiscard {
            game.fill_valid_actions(valid);
            assert!(game.execute_action(&valid[0]));
        }
        // Move the robber onto the occupied tile and steal.
        assert!(game.execute_action(&Action::MoveRobber { player: 0, tile: 9 }));
        assert_eq!(game.turn_phase, TurnPhase::RobberSteal);
        game.fill_valid_actions(valid);
        assert!(game.execute_action(&valid[0]));
        // Park the robber elsewhere so the next cycle can move it back.
        game.state.robber_tile = 0;
    };

    cycle(&mut game, &mut valid); // warmup: buffers reach capacity
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    cycle(&mut game, &mut valid);
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
    assert_eq!(
        allocations, 0,
        "{allocations} allocations during a steady-state robber cycle (7-roll, discards, move, steal)"
    );
}
