//! Every action is validated against the current phase and acting player,
//! and a rejected action must leave the game completely unchanged.

use catan_core::game::{Action, CatanGame, GamePhase, TurnPhase};

/// Everything observable about a game, minus the RNG (a rejected action must
/// never draw from it either — equal fingerprints prove that indirectly).
fn fingerprint(game: &CatanGame) -> impl PartialEq + std::fmt::Debug {
    let s = &game.state;
    (
        (
            s.vertices,
            s.edges,
            s.resources,
            s.bank,
            s.dev_cards,
            s.dev_deck_idx,
            s.knights_played,
        ),
        (
            s.settlements_built,
            s.cities_built,
            s.roads_built,
            s.robber_tile,
            s.dice_roll,
            s.has_rolled,
            s.turn,
            s.current_player,
            s.winner,
        ),
        (
            s.longest_road_player,
            s.longest_road_length,
            s.largest_army_player,
            s.largest_army_size,
            s.dev_cards_bought_this_turn,
            s.dev_card_played_this_turn,
        ),
        (
            game.game_phase as u8,
            game.turn_phase as u8,
            game.setup_player_idx,
            game.roads_to_place,
            game.discard_idx,
            game.trades_proposed_this_turn,
            game.action_history.len(),
        ),
    )
}

fn assert_rejected(game: &mut CatanGame, action: Action) {
    let before = fingerprint(game);
    assert!(
        !game.execute_action(&action),
        "{action:?} should be rejected"
    );
    assert_eq!(
        before,
        fingerprint(game),
        "{action:?} mutated state despite rejection"
    );
}

#[test]
fn setup_phase_rejects_all_playing_actions() {
    let mut game = CatanGame::new(4, 41);
    assert_eq!(game.game_phase, GamePhase::SetupForward);
    for action in [
        Action::RollDice {
            player: 0,
            forced: None,
        },
        Action::BuildRoad { player: 0, edge: 0 },
        Action::BuildSettlement {
            player: 0,
            vertex: 0,
        },
        Action::BuildCity {
            player: 0,
            vertex: 0,
        },
        Action::BuyDevCard { player: 0 },
        Action::PlayKnight { player: 0 },
        Action::EndTurn { player: 0 },
        Action::MoveRobber { player: 0, tile: 1 },
        Action::TradeWithBank {
            player: 0,
            give: 0,
            recv: 1,
        },
        Action::ProposeTrade {
            player: 0,
            give: 0,
            give_amount: 1,
            recv: 1,
        },
    ] {
        assert_rejected(&mut game, action);
    }
}

#[test]
fn setup_actions_validate_the_acting_player() {
    let mut game = CatanGame::new(4, 41);
    let valid = game.valid_actions();
    let Action::PlaceInitialSettlement { vertex, .. } = valid[0] else {
        panic!("setup starts with settlement placement");
    };
    // Right vertex, wrong player.
    assert_rejected(
        &mut game,
        Action::PlaceInitialSettlement { player: 1, vertex },
    );
    // Road before settlement.
    assert_rejected(&mut game, Action::PlaceInitialRoad { player: 0, edge: 0 });
}

#[test]
fn must_roll_phase_rejects_everything_but_rolling() {
    let mut game = CatanGame::new(4, 41);
    game.game_phase = GamePhase::Playing;
    game.state.phase = 1;
    game.turn_phase = TurnPhase::MustRoll;
    game.state.current_player = 2;
    game.state.resources[2] = [10, 10, 10, 10, 10];

    assert_rejected(&mut game, Action::EndTurn { player: 2 });
    assert_rejected(&mut game, Action::BuyDevCard { player: 2 });
    assert_rejected(
        &mut game,
        Action::TradeWithBank {
            player: 2,
            give: 0,
            recv: 1,
        },
    );
    // Wrong player can't roll.
    assert_rejected(
        &mut game,
        Action::RollDice {
            player: 0,
            forced: None,
        },
    );
    // Right player can.
    assert!(game.execute_action(&Action::RollDice {
        player: 2,
        forced: Some(4)
    }));
}

#[test]
fn main_phase_rejects_disconnected_builds() {
    let mut game = CatanGame::new(4, 41);
    game.game_phase = GamePhase::Playing;
    game.state.phase = 1;
    game.turn_phase = TurnPhase::Main;
    game.state.current_player = 0;
    game.state.has_rolled = true;
    game.state.resources[0] = [10, 10, 10, 10, 10];

    // No roads anywhere: any settlement build is disconnected, any road is too.
    assert_rejected(
        &mut game,
        Action::BuildSettlement {
            player: 0,
            vertex: 20,
        },
    );
    assert_rejected(
        &mut game,
        Action::BuildRoad {
            player: 0,
            edge: 30,
        },
    );
    // No settlement at vertex 20: city upgrade impossible.
    assert_rejected(
        &mut game,
        Action::BuildCity {
            player: 0,
            vertex: 20,
        },
    );
}

#[test]
fn robber_phases_validate_actor_and_target() {
    let mut game = CatanGame::new(4, 41);
    game.game_phase = GamePhase::Playing;
    game.state.phase = 1;
    game.state.current_player = 1;
    game.turn_phase = TurnPhase::RobberMove;

    // Robber can't stay put; mover must be the current player.
    let here = game.state.robber_tile;
    assert_rejected(
        &mut game,
        Action::MoveRobber {
            player: 1,
            tile: here,
        },
    );
    let target = (0..19u8).find(|&t| t != here).unwrap();
    assert_rejected(
        &mut game,
        Action::MoveRobber {
            player: 0,
            tile: target,
        },
    );
    // Steal is not available during RobberMove.
    assert_rejected(
        &mut game,
        Action::StealResource {
            player: 1,
            victim: 0,
            forced: None,
        },
    );
    assert!(game.execute_action(&Action::MoveRobber {
        player: 1,
        tile: target
    }));
}

#[test]
fn steal_target_must_be_stealable() {
    let mut game = CatanGame::new(4, 41);
    game.game_phase = GamePhase::Playing;
    game.state.phase = 1;
    game.state.current_player = 0;
    game.turn_phase = TurnPhase::RobberSteal;
    // Nobody on the robber tile: only the skip action (victim -1) is legal.
    assert_rejected(
        &mut game,
        Action::StealResource {
            player: 0,
            victim: 2,
            forced: None,
        },
    );
    assert!(game.execute_action(&Action::StealResource {
        player: 0,
        victim: -1,
        forced: None
    }));
    assert_eq!(game.turn_phase, TurnPhase::Main);
}

#[test]
fn every_legal_action_executes_successfully() {
    // Mask/execution consistency on real game states: anything the engine
    // offers must execute on a clone.
    for seed in [3u64, 17, 99] {
        let mut game = CatanGame::new(4, seed);
        let mut rng_choice = 0usize;
        for _step in 0..2000 {
            if game.is_game_over() {
                break;
            }
            let valid = game.valid_actions();
            if valid.is_empty() {
                break;
            }
            for action in &valid {
                let mut probe = game.clone();
                assert!(
                    probe.execute_action(action),
                    "engine offered {action:?} but rejected it (seed {seed})"
                );
            }
            // Advance deterministically through varied branches.
            rng_choice = (rng_choice * 31 + 7) % valid.len();
            game.execute_action(&valid[rng_choice]);
        }
    }
}
