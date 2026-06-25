//! Player-to-player trading: restricted offer menu (give 1-2 of one resource
//! for 1 of another), propose -> respond -> confirm flow, per-turn offer cap.

use catan_core::game::{Action, CatanGame, GamePhase, TurnPhase};

const WHEAT: u8 = 0;
const SHEEP: u8 = 1;
const WOOD: u8 = 2;

/// Drive a fresh game directly into player 0's main phase with chosen hands.
fn main_phase_game(hands: [[i16; 5]; 4]) -> CatanGame {
    let mut game = CatanGame::new(4, 11);
    game.game_phase = GamePhase::Playing;
    game.state.phase = 1;
    game.turn_phase = TurnPhase::Main;
    game.state.current_player = 0;
    game.state.has_rolled = true;
    for p in 0..4 {
        for r in 0..5 {
            // Move resources from the bank so conservation holds.
            game.state.resources[p][r] = hands[p][r];
            game.state.bank[r] -= hands[p][r];
        }
    }
    game
}

fn propose(give: u8, amount: u8, recv: u8) -> Action {
    Action::ProposeTrade {
        player: 0,
        give,
        give_amount: amount,
        recv,
    }
}

#[test]
fn propose_requires_holdings_and_main_phase() {
    let mut game = main_phase_game([[2, 0, 0, 0, 0], [0; 5], [0; 5], [0; 5]]);
    // Has 2 wheat: 1-for-1 and 2-for-1 wheat offers are legal.
    let valid = game.valid_actions();
    assert!(valid.contains(&propose(WHEAT, 1, SHEEP)));
    assert!(valid.contains(&propose(WHEAT, 2, SHEEP)));
    // No sheep: cannot offer sheep.
    assert!(!valid.contains(&propose(SHEEP, 1, WHEAT)));
    // Cannot offer a resource for itself.
    assert!(!valid.contains(&propose(WHEAT, 1, WHEAT)));
    // Executing an unaffordable offer is rejected.
    assert!(!game.execute_action(&propose(SHEEP, 1, WHEAT)));
}

#[test]
fn responders_who_cannot_afford_are_skipped() {
    // Only player 2 holds the requested sheep; players 1 and 3 are skipped.
    let mut game = main_phase_game([[2, 0, 0, 0, 0], [0; 5], [0, 3, 0, 0, 0], [0; 5]]);
    assert!(game.execute_action(&propose(WHEAT, 1, SHEEP)));
    assert_eq!(game.turn_phase, TurnPhase::TradeResponse);
    assert_eq!(game.current_player(), 2, "only eligible responder is asked");
}

#[test]
fn offer_with_no_eligible_responder_returns_to_main() {
    let mut game = main_phase_game([[2, 0, 0, 0, 0], [0; 5], [0; 5], [0; 5]]);
    assert!(game.execute_action(&propose(WHEAT, 1, SHEEP)));
    assert_eq!(
        game.turn_phase,
        TurnPhase::Main,
        "nobody can accept: offer fizzles"
    );
}

#[test]
fn full_trade_flow_exchanges_resources() {
    let mut game = main_phase_game([[2, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0; 5]]);
    assert!(game.execute_action(&propose(WHEAT, 2, SHEEP)));

    // Responders are asked in seat order after the proposer.
    assert_eq!(game.current_player(), 1);
    let valid = game.valid_actions();
    assert!(valid.contains(&Action::RespondTrade {
        player: 1,
        accept: true
    }));
    assert!(valid.contains(&Action::RespondTrade {
        player: 1,
        accept: false
    }));
    assert!(game.execute_action(&Action::RespondTrade {
        player: 1,
        accept: true
    }));

    assert_eq!(game.current_player(), 2);
    assert!(game.execute_action(&Action::RespondTrade {
        player: 2,
        accept: true
    }));

    // Both accepted: proposer chooses a partner (or cancels).
    assert_eq!(game.turn_phase, TurnPhase::TradeChoose);
    assert_eq!(game.current_player(), 0);
    let valid = game.valid_actions();
    assert!(valid.contains(&Action::ConfirmTrade {
        player: 0,
        partner: 1
    }));
    assert!(valid.contains(&Action::ConfirmTrade {
        player: 0,
        partner: 2
    }));
    assert!(valid.contains(&Action::ConfirmTrade {
        player: 0,
        partner: -1
    }));
    // Confirming a non-acceptor is illegal.
    assert!(!game.execute_action(&Action::ConfirmTrade {
        player: 0,
        partner: 3
    }));

    assert!(game.execute_action(&Action::ConfirmTrade {
        player: 0,
        partner: 2
    }));
    assert_eq!(game.turn_phase, TurnPhase::Main);
    // 2 wheat went to player 2; 1 sheep came back.
    assert_eq!(game.state.resources[0][0], 0);
    assert_eq!(game.state.resources[0][1], 1);
    assert_eq!(game.state.resources[2][0], 2);
    assert_eq!(game.state.resources[2][1], 0);
    // Player 1 (unchosen acceptor) is untouched.
    assert_eq!(game.state.resources[1][1], 1);
}

#[test]
fn cancel_leaves_everyone_unchanged() {
    let mut game = main_phase_game([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0; 5], [0; 5]]);
    assert!(game.execute_action(&propose(WHEAT, 1, SHEEP)));
    assert!(game.execute_action(&Action::RespondTrade {
        player: 1,
        accept: true
    }));
    assert!(game.execute_action(&Action::ConfirmTrade {
        player: 0,
        partner: -1
    }));
    assert_eq!(game.turn_phase, TurnPhase::Main);
    assert_eq!(game.state.resources[0][0], 1);
    assert_eq!(game.state.resources[1][1], 1);
}

#[test]
fn all_rejections_return_to_main_without_exchange() {
    let mut game = main_phase_game([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0; 5]]);
    assert!(game.execute_action(&propose(WHEAT, 1, SHEEP)));
    assert!(game.execute_action(&Action::RespondTrade {
        player: 1,
        accept: false
    }));
    assert!(game.execute_action(&Action::RespondTrade {
        player: 2,
        accept: false
    }));
    assert_eq!(game.turn_phase, TurnPhase::Main);
    assert_eq!(game.state.resources[0][0], 1, "no exchange happened");
}

#[test]
fn per_turn_offer_cap_is_enforced_and_resets() {
    let mut game = main_phase_game([[9, 0, 0, 0, 0], [0; 5], [0; 5], [0; 5]]);
    for _ in 0..3 {
        // No eligible responders, so each offer fizzles back to Main.
        assert!(game.execute_action(&propose(WHEAT, 1, SHEEP)));
    }
    // Cap reached: proposals disappear from the action list and are rejected.
    let valid = game.valid_actions();
    assert!(!valid
        .iter()
        .any(|a| matches!(a, Action::ProposeTrade { .. })));
    assert!(!game.execute_action(&propose(WHEAT, 1, SHEEP)));

    // New turn: cap resets.
    assert!(game.execute_action(&Action::EndTurn { player: 0 }));
    game.state.current_player = 0; // pretend the table came back around
    game.turn_phase = TurnPhase::Main;
    let valid = game.valid_actions();
    assert!(valid
        .iter()
        .any(|a| matches!(a, Action::ProposeTrade { .. })));
}

#[test]
fn responder_actions_are_validated() {
    let mut game = main_phase_game([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0; 5]]);
    assert!(game.execute_action(&propose(WHEAT, 1, SHEEP)));
    assert_eq!(game.current_player(), 1);
    // Player 2 may not answer out of turn; proposer may not answer at all.
    assert!(!game.execute_action(&Action::RespondTrade {
        player: 2,
        accept: true
    }));
    assert!(!game.execute_action(&Action::RespondTrade {
        player: 0,
        accept: true
    }));
    // Building mid-negotiation is illegal.
    assert!(!game.execute_action(&Action::EndTurn { player: 0 }));
}

#[test]
fn trades_conserve_resources() {
    let mut game = main_phase_game([[2, 0, 1, 0, 0], [0, 2, 0, 0, 0], [0; 5], [0; 5]]);
    let total_before: i16 = (0..5)
        .map(|r| game.state.bank[r] + (0..4).map(|p| game.state.resources[p][r]).sum::<i16>())
        .sum();
    assert!(game.execute_action(&propose(WOOD, 1, SHEEP)));
    assert!(game.execute_action(&Action::RespondTrade {
        player: 1,
        accept: true
    }));
    assert!(game.execute_action(&Action::ConfirmTrade {
        player: 0,
        partner: 1
    }));
    let total_after: i16 = (0..5)
        .map(|r| game.state.bank[r] + (0..4).map(|p| game.state.resources[p][r]).sum::<i16>())
        .sum();
    assert_eq!(total_before, total_after);
}
