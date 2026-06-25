//! The action codec is the contract between engine and policy network.
//! These tests pin it three ways:
//! 1. Layout spot checks: known actions map to their documented ids.
//! 2. Roundtrip + uniqueness on live states: every legal action encodes to
//!    a distinct id and decodes back to itself.
//! 3. Mask <-> engine equivalence fuzz: for EVERY id at sampled states,
//!    mask[id] is true exactly when executing decode(id) succeeds — the
//!    mask can never lie to the network.

use catan_core::game::{Action, CatanGame, GamePhase, TurnPhase};
use catan_core::players::{Player, RandomPlayer};
use catan_env::{decode_action, encode_action, fill_action_mask, NUM_ACTIONS};

#[test]
fn layout_spot_checks() {
    let game = CatanGame::new(4, 1); // setup phase, actor = player 0
    let enc = |a: &Action| encode_action(&game, a);

    assert_eq!(
        enc(&Action::PlaceInitialSettlement {
            player: 0,
            vertex: 10
        }),
        10
    );
    assert_eq!(
        enc(&Action::BuildSettlement {
            player: 0,
            vertex: 10
        }),
        10,
        "shared id"
    );
    assert_eq!(
        enc(&Action::BuildCity {
            player: 0,
            vertex: 0
        }),
        54
    );
    assert_eq!(
        enc(&Action::PlaceInitialRoad {
            player: 0,
            edge: 71
        }),
        108 + 71
    );
    assert_eq!(
        enc(&Action::MoveRobber {
            player: 0,
            tile: 18
        }),
        180 + 18
    );
    // Steal: relative seats. Actor 0 stealing from seat 1 = rel 1.
    assert_eq!(
        enc(&Action::StealResource {
            player: 0,
            victim: 1,
            forced: None
        }),
        199
    );
    assert_eq!(
        enc(&Action::StealResource {
            player: 0,
            victim: -1,
            forced: None
        }),
        202
    );
    assert_eq!(
        enc(&Action::DiscardResource {
            player: 0,
            resource: 4
        }),
        203 + 4
    );
    assert_eq!(
        enc(&Action::PlayMonopoly {
            player: 0,
            resource: 0
        }),
        208
    );
    assert_eq!(
        enc(&Action::PlayYearOfPlenty {
            player: 0,
            r1: 0,
            r2: 0
        }),
        213
    );
    assert_eq!(
        enc(&Action::PlayYearOfPlenty {
            player: 0,
            r1: 4,
            r2: 4
        }),
        213 + 14
    );
    assert_eq!(
        enc(&Action::TradeWithBank {
            player: 0,
            give: 0,
            recv: 1
        }),
        228
    );
    assert_eq!(
        enc(&Action::TradeWithBank {
            player: 0,
            give: 4,
            recv: 3
        }),
        228 + 19
    );
    assert_eq!(
        enc(&Action::ProposeTrade {
            player: 0,
            give: 0,
            give_amount: 1,
            recv: 1
        }),
        248
    );
    assert_eq!(
        enc(&Action::ProposeTrade {
            player: 0,
            give: 4,
            give_amount: 2,
            recv: 3
        }),
        248 + 39
    );
    assert_eq!(
        enc(&Action::RespondTrade {
            player: 0,
            accept: true
        }),
        288
    );
    assert_eq!(
        enc(&Action::RespondTrade {
            player: 0,
            accept: false
        }),
        289
    );
    assert_eq!(
        enc(&Action::ConfirmTrade {
            player: 0,
            partner: 1
        }),
        290
    );
    assert_eq!(
        enc(&Action::ConfirmTrade {
            player: 0,
            partner: -1
        }),
        293
    );
    assert_eq!(
        enc(&Action::RollDice {
            player: 0,
            forced: None
        }),
        294
    );
    assert_eq!(enc(&Action::BuyDevCard { player: 0 }), 295);
    assert_eq!(enc(&Action::PlayKnight { player: 0 }), 296);
    assert_eq!(enc(&Action::PlayRoadBuilding { player: 0 }), 297);
    assert_eq!(enc(&Action::EndTurn { player: 0 }), 298);
}

#[test]
fn seat_relative_encoding_rotates_with_the_actor() {
    // The same id must mean "the seat to my left" for every actor.
    let mut game = CatanGame::new(4, 2);
    game.game_phase = GamePhase::Playing;
    game.state.phase = 1;
    game.turn_phase = TurnPhase::RobberSteal;
    for actor in 0..4usize {
        game.state.current_player = actor;
        let left = (actor + 1) % 4;
        let id = encode_action(
            &game,
            &Action::StealResource {
                player: actor as u8,
                victim: left as i8,
                forced: None,
            },
        );
        assert_eq!(
            id, 199,
            "steal-left is id 199 for every actor (actor {actor})"
        );
        let decoded = decode_action(&game, 199);
        assert_eq!(
            decoded,
            Action::StealResource {
                player: actor as u8,
                victim: left as i8,
                forced: None
            }
        );
    }
}

/// Drive games with random players; at every step check roundtrip and id
/// uniqueness, and at sampled steps check full mask <-> engine equivalence.
fn fuzz_codec(seed: u64, num_players: usize) {
    let mut game = CatanGame::new(num_players, seed);
    game.record_history = false;
    let mut players: Vec<RandomPlayer> = (0..num_players as u64)
        .map(|i| RandomPlayer::new(seed * 13 + i))
        .collect();

    let mut valid = Vec::with_capacity(512);
    let mut scratch = Vec::with_capacity(512);
    let mut mask = [false; NUM_ACTIONS];
    let mut steps = 0u32;
    let mut equivalence_checks = 0u32;

    while !game.is_game_over() && game.state.turn < 1000 {
        game.fill_valid_actions(&mut valid);
        if valid.is_empty() {
            break;
        }

        // Roundtrip + uniqueness for every legal action.
        let mut seen = [false; NUM_ACTIONS];
        for action in &valid {
            let id = encode_action(&game, action);
            assert!(id < NUM_ACTIONS, "id out of range for {action:?}");
            assert!(!seen[id], "seed {seed}: id {id} used twice in one mask");
            seen[id] = true;
            let decoded = decode_action(&game, id);
            assert_eq!(
                decoded, *action,
                "seed {seed}: roundtrip failed for id {id}"
            );
        }

        // Mask must be exactly the encode-image of the valid set.
        fill_action_mask(&game, &mut scratch, &mut mask);
        assert_eq!(
            mask, seen,
            "seed {seed}: mask differs from encoded valid set"
        );

        // Sampled full equivalence: every id executes iff masked legal.
        if steps.is_multiple_of(53) {
            for id in 0..NUM_ACTIONS {
                let action = decode_action(&game, id);
                let mut probe = game.clone();
                let executed = probe.execute_action(&action);
                assert_eq!(
                    executed, mask[id],
                    "seed {seed} step {steps}: id {id} ({action:?}) executed={executed} \
                     but mask says {}",
                    mask[id]
                );
            }
            equivalence_checks += 1;
        }

        let idx = game.current_player();
        let action = players[idx].choose_action(&game, &valid);
        game.execute_action(&action);
        steps += 1;
    }
    assert!(
        equivalence_checks >= 10,
        "seed {seed}: too few equivalence samples"
    );
    println!(
        "seed {seed} ({num_players}p): {steps} steps, {equivalence_checks} states x {NUM_ACTIONS} ids equivalence-checked"
    );
}

#[test]
fn mask_matches_engine_exactly_4p() {
    for seed in [1u64, 9, 23] {
        fuzz_codec(seed, 4);
    }
}

#[test]
fn mask_matches_engine_exactly_3p() {
    // 3-player games: the rel-seat-3 ids must never be legal and must
    // always be rejected (they wrap onto the actor).
    for seed in [5u64, 14] {
        fuzz_codec(seed, 3);
    }
}

#[test]
fn decode_is_total_in_every_phase() {
    // decode must return some action for every id without panicking, in
    // states drawn from every reachable phase.
    let mut game = CatanGame::new(4, 31);
    game.record_history = false;
    let mut players: Vec<RandomPlayer> = (0..4u64).map(|i| RandomPlayer::new(77 + i)).collect();
    let mut valid = Vec::with_capacity(512);
    let mut phases_seen = std::collections::HashSet::new();

    while !game.is_game_over() && game.state.turn < 400 {
        phases_seen.insert((game.game_phase as u8, game.turn_phase as u8));
        for id in 0..NUM_ACTIONS {
            let _ = decode_action(&game, id);
        }
        game.fill_valid_actions(&mut valid);
        if valid.is_empty() {
            break;
        }
        let idx = game.current_player();
        let action = players[idx].choose_action(&game, &valid);
        game.execute_action(&action);
    }
    assert!(
        phases_seen.len() >= 8,
        "only {} distinct phases reached — fuzz too shallow",
        phases_seen.len()
    );
}
