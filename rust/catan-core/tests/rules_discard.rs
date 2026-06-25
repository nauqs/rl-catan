//! Discarding on a 7: players over 7 cards discard half (rounded down), one
//! card at a time, chosen by the discarding player.

use catan_core::game::{Action, CatanGame, GamePhase, TurnPhase};

fn rolling_game(hands: [[i16; 5]; 4]) -> CatanGame {
    let mut game = CatanGame::new(4, 31);
    game.game_phase = GamePhase::Playing;
    game.state.phase = 1;
    game.turn_phase = TurnPhase::MustRoll;
    game.state.current_player = 0;
    for p in 0..4 {
        for r in 0..5 {
            game.state.resources[p][r] = hands[p][r];
            game.state.bank[r] -= hands[p][r];
        }
    }
    game
}

#[test]
fn seven_forces_each_over_limit_player_to_discard_half() {
    // Player 1: 8 cards -> 4; player 3: 9 cards -> 4; players 0/2 safe.
    let mut game = rolling_game([
        [3, 3, 0, 0, 0],
        [8, 0, 0, 0, 0],
        [7, 0, 0, 0, 0],
        [3, 3, 3, 0, 0],
    ]);
    assert!(game.execute_action(&Action::RollDice {
        player: 0,
        forced: Some(7)
    }));
    assert_eq!(game.turn_phase, TurnPhase::RobberDiscard);

    // Player 1 is asked first and must discard 4 wheat one at a time.
    assert_eq!(game.current_player(), 1);
    for _ in 0..4 {
        let valid = game.valid_actions();
        assert_eq!(
            valid,
            vec![Action::DiscardResource {
                player: 1,
                resource: 0
            }],
            "only held resources are discardable"
        );
        assert!(game.execute_action(&valid[0]));
    }
    assert_eq!(game.state.resources[1][0], 4);

    // Then player 3, who can choose among held types.
    assert_eq!(game.current_player(), 3);
    let valid = game.valid_actions();
    assert_eq!(valid.len(), 3, "one discard action per held resource type");
    for _ in 0..4 {
        let valid = game.valid_actions();
        assert!(game.execute_action(&valid[0]));
    }
    assert_eq!(game.state.total_resources(3), 5);

    // All discards done: roller moves the robber.
    assert_eq!(game.turn_phase, TurnPhase::RobberMove);
    assert_eq!(game.current_player(), 0);
}

#[test]
fn discard_quota_is_fixed_at_roll_time() {
    // 9 cards -> discard 4 (floor), keeping 5.
    let mut game = rolling_game([[0; 5], [2, 2, 2, 2, 1], [0; 5], [0; 5]]);
    assert!(game.execute_action(&Action::RollDice {
        player: 0,
        forced: Some(7)
    }));
    let mut discards = 0;
    while game.turn_phase == TurnPhase::RobberDiscard {
        let valid = game.valid_actions();
        assert!(game.execute_action(&valid[0]));
        discards += 1;
    }
    assert_eq!(discards, 4);
    assert_eq!(game.state.total_resources(1), 5);
}

#[test]
fn discards_return_to_bank() {
    let mut game = rolling_game([[0; 5], [8, 0, 0, 0, 0], [0; 5], [0; 5]]);
    let bank_wheat = game.state.bank[0];
    assert!(game.execute_action(&Action::RollDice {
        player: 0,
        forced: Some(7)
    }));
    while game.turn_phase == TurnPhase::RobberDiscard {
        let valid = game.valid_actions();
        assert!(game.execute_action(&valid[0]));
    }
    assert_eq!(game.state.bank[0], bank_wheat + 4);
}

#[test]
fn cannot_discard_unheld_resource_or_out_of_turn() {
    let mut game = rolling_game([[0; 5], [8, 0, 0, 0, 0], [9, 0, 0, 0, 0], [0; 5]]);
    assert!(game.execute_action(&Action::RollDice {
        player: 0,
        forced: Some(7)
    }));
    assert_eq!(game.current_player(), 1);
    // Unheld resource type.
    assert!(!game.execute_action(&Action::DiscardResource {
        player: 1,
        resource: 3
    }));
    // Wrong player (player 2 also owes, but it's not their turn to discard).
    assert!(!game.execute_action(&Action::DiscardResource {
        player: 2,
        resource: 0
    }));
    // Roller can't move the robber before discards finish.
    assert!(!game.execute_action(&Action::MoveRobber { player: 0, tile: 0 }));
}

#[test]
fn exactly_seven_cards_is_safe() {
    let mut game = rolling_game([[0; 5], [7, 0, 0, 0, 0], [0; 5], [0; 5]]);
    assert!(game.execute_action(&Action::RollDice {
        player: 0,
        forced: Some(7)
    }));
    assert_eq!(
        game.turn_phase,
        TurnPhase::RobberMove,
        "7 cards exactly: no discard"
    );
}
