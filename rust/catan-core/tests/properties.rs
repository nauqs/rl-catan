//! Property-based testing: random games must uphold every invariant at every
//! step; the fast longest-road must match a brute-force oracle; and any
//! action outside the legal set must be rejected without side effects.

use std::collections::HashSet;

use proptest::prelude::*;

use catan_core::board::topology;
use catan_core::game::{Action, CatanGame, GamePhase};
use catan_core::players::{Player, RandomPlayer};
use catan_core::state::GameState;

// ---------------------------------------------------------------- invariants

fn check_invariants(game: &CatanGame, ctx: &str) {
    let state = &game.state;
    let n = state.num_players;

    for r in 0..5 {
        let held: i16 = (0..n).map(|p| state.resources[p][r]).sum();
        assert_eq!(
            state.bank[r] + held,
            19,
            "{ctx}: resource {r} not conserved"
        );
        assert!(state.bank[r] >= 0, "{ctx}: bank negative");
        for p in 0..n {
            assert!(
                state.resources[p][r] >= 0,
                "{ctx}: player {p} negative resources"
            );
        }
    }

    for p in 0..n {
        let s = state.vertices.iter().filter(|&&v| v == p as i8).count();
        let c = state.vertices.iter().filter(|&&v| v == p as i8 + 4).count();
        let r = state.edges.iter().filter(|&&e| e == p as i8).count();
        assert_eq!(
            state.settlements_built[p] as usize, s,
            "{ctx}: settlement counter"
        );
        assert_eq!(state.cities_built[p] as usize, c, "{ctx}: city counter");
        assert_eq!(state.roads_built[p] as usize, r, "{ctx}: road counter");
        assert!(s <= 5 && c <= 4 && r <= 15, "{ctx}: building limits");
    }

    // Distance rule holds globally.
    let topo = topology();
    for v in 0..54 {
        if state.vertices[v] >= 0 {
            for &nb in &topo.vertex_neighbors[v] {
                assert!(
                    nb < 0 || state.vertices[nb as usize] < 0,
                    "{ctx}: distance rule violated at {v}/{nb}"
                );
            }
        }
    }

    assert!((state.robber_tile as usize) < 19, "{ctx}: robber off board");

    // Dev card accounting.
    let drawn = state.dev_deck_idx as i32;
    let held: i32 = (0..n)
        .map(|p| state.dev_cards[p].iter().map(|&c| c as i32).sum::<i32>())
        .sum();
    let knights: i32 = (0..n).map(|p| state.knights_played[p] as i32).sum();
    assert!(drawn - held - knights >= 0, "{ctx}: dev cards over-issued");

    // Winner consistency.
    if game.game_phase == GamePhase::Finished {
        let w = state.winner;
        assert!(w >= 0, "{ctx}: finished without winner");
        assert!(
            state.calculate_victory_points(w as usize) >= 10,
            "{ctx}: winner below 10 VP"
        );
    }
}

// ------------------------------------------------------- longest-road oracle

/// Brute-force longest trail: walk the player's road subgraph, never reusing
/// an edge, never continuing through an opponent-occupied vertex.
fn oracle_longest_trail(state: &GameState, player: usize) -> usize {
    let topo = topology();
    let player_i8 = player as i8;

    fn walk(state: &GameState, player_i8: i8, at: u8, used: u128) -> usize {
        let topo = topology();
        let owner = state.settlement_owner(at as usize);
        if owner >= 0 && owner != player_i8 {
            return 0; // can't continue through an opponent's building
        }
        let mut best = 0;
        for &adj in &topo.vertex_edges[at as usize] {
            if adj >= 0 {
                let e = adj as usize;
                if state.edges[e] == player_i8 && used & (1u128 << e) == 0 {
                    let [a, b] = topo.edge_vertices[e];
                    let next = if a == at { b } else { a };
                    best = best.max(1 + walk(state, player_i8, next, used | (1u128 << e)));
                }
            }
        }
        best
    }

    let mut best = 0;
    for e in 0..72usize {
        if state.edges[e] == player_i8 {
            let [a, b] = topo.edge_vertices[e];
            best = best.max(1 + walk(state, player_i8, b, 1u128 << e));
            best = best.max(1 + walk(state, player_i8, a, 1u128 << e));
        }
    }
    best
}

fn check_longest_road_oracle(state: &GameState, ctx: &str) {
    for p in 0..state.num_players {
        let fast = catan_core::building::longest_road_length(state, p);
        let count = state.edges.iter().filter(|&&e| e == p as i8).count();
        if count >= 5 {
            let oracle = oracle_longest_trail(state, p);
            assert_eq!(fast, oracle, "{ctx}: longest road mismatch for player {p}");
        } else {
            assert_eq!(fast, count, "{ctx}: sub-5 edge count rule");
        }
    }
}

// ------------------------------------------------- illegal action generation

/// Every parameterization of every action variant, for every seat.
fn action_universe(num_players: usize) -> Vec<Action> {
    let mut universe = Vec::new();
    for p in 0..num_players as u8 {
        for v in 0..54u8 {
            universe.push(Action::PlaceInitialSettlement {
                player: p,
                vertex: v,
            });
            universe.push(Action::BuildSettlement {
                player: p,
                vertex: v,
            });
            universe.push(Action::BuildCity {
                player: p,
                vertex: v,
            });
        }
        for e in 0..72u8 {
            universe.push(Action::PlaceInitialRoad { player: p, edge: e });
            universe.push(Action::BuildRoad { player: p, edge: e });
        }
        for t in 0..19u8 {
            universe.push(Action::MoveRobber { player: p, tile: t });
        }
        for victim in -1..num_players as i8 {
            universe.push(Action::StealResource {
                player: p,
                victim,
                forced: None,
            });
        }
        for r in 0..5u8 {
            universe.push(Action::DiscardResource {
                player: p,
                resource: r,
            });
            universe.push(Action::PlayMonopoly {
                player: p,
                resource: r,
            });
            for r2 in r..5u8 {
                universe.push(Action::PlayYearOfPlenty {
                    player: p,
                    r1: r,
                    r2,
                });
            }
            for r2 in 0..5u8 {
                if r != r2 {
                    universe.push(Action::TradeWithBank {
                        player: p,
                        give: r,
                        recv: r2,
                    });
                    for amount in 1..=2u8 {
                        universe.push(Action::ProposeTrade {
                            player: p,
                            give: r,
                            give_amount: amount,
                            recv: r2,
                        });
                    }
                }
            }
        }
        for accept in [false, true] {
            universe.push(Action::RespondTrade { player: p, accept });
        }
        for partner in -1..num_players as i8 {
            universe.push(Action::ConfirmTrade { player: p, partner });
        }
        universe.push(Action::RollDice {
            player: p,
            forced: None,
        });
        universe.push(Action::BuyDevCard { player: p });
        universe.push(Action::PlayKnight { player: p });
        universe.push(Action::PlayRoadBuilding { player: p });
        universe.push(Action::EndTurn { player: p });
    }
    universe
}

fn fingerprint(game: &CatanGame) -> String {
    let s = &game.state;
    format!(
        "{:?}|{:?}|{:?}|{:?}|{:?}|{}|{:?}|{:?}|{}|{}|{}|{}|{:?}|{:?}|{:?}|{:?}|{}|{}|{:?}|{}|{}|{}",
        s.vertices,
        s.edges,
        s.resources,
        s.bank,
        s.dev_cards,
        s.dev_deck_idx,
        s.knights_played,
        s.settlements_built,
        s.robber_tile,
        s.dice_roll,
        s.has_rolled,
        s.turn,
        s.cities_built,
        s.roads_built,
        (s.longest_road_player, s.longest_road_length),
        (s.largest_army_player, s.largest_army_size),
        s.current_player,
        s.winner,
        (game.game_phase as u8, game.turn_phase as u8),
        game.setup_player_idx,
        game.roads_to_place,
        game.action_history.len(),
    )
}

/// Returns (legal actions verified executable, illegal actions verified
/// rejected without side effects).
fn fuzz_illegal_actions(game: &mut CatanGame, universe: &[Action], ctx: &str) -> (usize, usize) {
    let legal: HashSet<Action> = game.valid_actions().into_iter().collect();
    // Legal actions must execute on a clone.
    for action in &legal {
        let mut probe = game.clone();
        assert!(
            probe.execute_action(action),
            "{ctx}: offered {action:?} but rejected"
        );
    }
    // Everything else must be rejected with zero side effects.
    let mut illegal = 0;
    for action in universe {
        if legal.contains(action) {
            continue;
        }
        let before = fingerprint(game);
        assert!(
            !game.execute_action(action),
            "{ctx}: {action:?} not in legal set but executed"
        );
        assert_eq!(
            before,
            fingerprint(game),
            "{ctx}: rejected {action:?} mutated state"
        );
        illegal += 1;
    }
    (legal.len(), illegal)
}

// -------------------------------------------------------------------- drivers

#[derive(Default)]
struct CheckStats {
    games: usize,
    steps: usize,
    invariant_checks: usize,
    oracle_checks: usize,
    fuzzed_states: usize,
    legal_verified: usize,
    illegal_verified: usize,
}

impl CheckStats {
    fn report(&self, label: &str) {
        // Visible with `cargo test -- --nocapture` (cargo hides it on success).
        println!(
            "[{label}] {} games, {} actions executed, {} invariant sweeps, \
             {} oracle cross-checks, {} states fuzzed \
             ({} legal actions verified executable, {} illegal verified rejected)",
            self.games,
            self.steps,
            self.invariant_checks,
            self.oracle_checks,
            self.fuzzed_states,
            self.legal_verified,
            self.illegal_verified,
        );
    }
}

fn run_checked_game(seed: u64, fuzz: bool, stats: &mut CheckStats) {
    let num_players = if seed.is_multiple_of(5) { 3 } else { 4 };
    let universe = action_universe(num_players);
    let mut game = CatanGame::new(num_players, seed);
    let mut players: Vec<RandomPlayer> = (0..num_players)
        .map(|i| RandomPlayer::new(seed * 31 + i as u64))
        .collect();

    stats.games += 1;
    let mut steps = 0u32;
    while !game.is_game_over() && game.state.turn < 1000 {
        let valid = game.valid_actions();
        assert!(
            !valid.is_empty(),
            "seed {seed}: no legal actions at turn {} phase {:?}",
            game.state.turn,
            game.turn_phase
        );
        let idx = game.current_player();
        let action = players[idx].choose_action(&game, &valid);
        assert!(
            game.execute_action(&action),
            "seed {seed}: chosen action rejected"
        );
        steps += 1;
        stats.steps += 1;

        check_invariants(&game, &format!("seed {seed} step {steps}"));
        stats.invariant_checks += 1;
        if steps.is_multiple_of(64) {
            check_longest_road_oracle(&game.state, &format!("seed {seed} step {steps}"));
            stats.oracle_checks += 1;
            if fuzz {
                let (legal, illegal) = fuzz_illegal_actions(
                    &mut game,
                    &universe,
                    &format!("seed {seed} step {steps}"),
                );
                stats.fuzzed_states += 1;
                stats.legal_verified += legal;
                stats.illegal_verified += illegal;
            }
        }
    }
    check_invariants(&game, &format!("seed {seed} final"));
    check_longest_road_oracle(&game.state, &format!("seed {seed} final"));
    stats.invariant_checks += 1;
    stats.oracle_checks += 1;
}

#[test]
fn random_games_uphold_all_invariants() {
    let mut stats = CheckStats::default();
    for seed in 0..25u64 {
        run_checked_game(seed, false, &mut stats);
    }
    stats.report("invariants");
}

#[test]
fn illegal_actions_are_always_rejected_without_side_effects() {
    let mut stats = CheckStats::default();
    for seed in [1u64, 8, 21] {
        run_checked_game(seed, true, &mut stats);
    }
    stats.report("fuzz");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Arbitrary choice streams (not just uniform-random players) still
    /// uphold every invariant — shrinking gives minimal failing prefixes.
    #[test]
    fn arbitrary_action_choices_uphold_invariants(
        seed in 0u64..10_000,
        choices in proptest::collection::vec(0usize..1024, 200..600),
    ) {
        let mut game = CatanGame::new(4, seed);
        for (i, &c) in choices.iter().enumerate() {
            if game.is_game_over() || game.state.turn >= 300 {
                break;
            }
            let valid = game.valid_actions();
            prop_assert!(!valid.is_empty(), "no legal actions mid-game");
            let action = valid[c % valid.len()];
            prop_assert!(game.execute_action(&action));
            if i % 16 == 0 {
                check_invariants(&game, &format!("proptest seed {seed} step {i}"));
            }
        }
        check_invariants(&game, &format!("proptest seed {seed} end"));
        check_longest_road_oracle(&game.state, &format!("proptest seed {seed} end"));
    }
}
