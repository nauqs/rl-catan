//! Development cards: buying, play restrictions, monopoly, year of plenty,
//! largest army.

use catan_core::dev_cards::{
    buy_dev_card, can_play_dev_card, play_knight, play_monopoly, play_year_of_plenty,
};
use catan_core::state::{
    GameState, DEV_CARD_COST, DEV_KNIGHT, DEV_MONOPOLY, DEV_VICTORY_POINT, DEV_YEAR_OF_PLENTY,
};

fn rich_state() -> GameState {
    let mut state = GameState::new(4, 99);
    state.resources[0] = [10, 10, 10, 10, 10];
    state
}

#[test]
fn buying_deducts_cost_and_draws_from_deck() {
    let mut state = rich_state();
    let deck_top = state.dev_deck[0];
    let card = buy_dev_card(&mut state, 0);
    assert_eq!(card, deck_top as i32);
    assert_eq!(state.dev_deck_idx, 1);
    assert_eq!(state.dev_cards[0][deck_top as usize], 1);
    for r in 0..5 {
        assert_eq!(state.resources[0][r], 10 - DEV_CARD_COST[r]);
    }
}

#[test]
fn deck_exhaustion_blocks_purchase() {
    let mut state = rich_state();
    state.dev_deck_idx = 25; // deck is 25 cards
    assert_eq!(buy_dev_card(&mut state, 0), -1);
}

#[test]
fn cannot_play_card_bought_this_turn() {
    let mut state = rich_state();
    let card = buy_dev_card(&mut state, 0);
    if card != DEV_VICTORY_POINT as i32 {
        assert!(!can_play_dev_card(&state, 0, card as usize));
    }
    // After the bought-this-turn marker clears, playable (except VP cards).
    state.dev_cards_bought_this_turn = [0; 5];
    if card != DEV_VICTORY_POINT as i32 {
        assert!(can_play_dev_card(&state, 0, card as usize));
    }
}

#[test]
fn victory_point_cards_are_never_playable_but_count() {
    let mut state = rich_state();
    state.dev_cards[0][DEV_VICTORY_POINT] = 2;
    assert!(!can_play_dev_card(&state, 0, DEV_VICTORY_POINT));
    assert_eq!(state.calculate_victory_points(0), 2);
}

#[test]
fn only_one_dev_card_per_turn() {
    let mut state = rich_state();
    state.dev_cards[0][DEV_KNIGHT] = 2;
    assert!(play_knight(&mut state, 0));
    assert!(
        !play_knight(&mut state, 0),
        "second card same turn must fail"
    );
}

#[test]
fn monopoly_takes_all_of_one_resource() {
    let mut state = rich_state();
    state.dev_cards[0][DEV_MONOPOLY] = 1;
    state.resources[1][2] = 3;
    state.resources[2][2] = 5;
    state.resources[3][2] = 0;
    let before = state.resources[0][2];
    let taken = play_monopoly(&mut state, 0, 2);
    assert_eq!(taken, 8);
    assert_eq!(state.resources[0][2], before + 8);
    assert_eq!(state.resources[1][2], 0);
    assert_eq!(state.resources[2][2], 0);
}

#[test]
fn year_of_plenty_respects_bank() {
    let mut state = rich_state();
    state.dev_cards[0][DEV_YEAR_OF_PLENTY] = 1;
    state.bank[3] = 1;
    assert!(
        !play_year_of_plenty(&mut state, 0, 3, 3),
        "bank lacks 2 brick"
    );
    assert!(
        !state.dev_card_played_this_turn,
        "failed play must not consume the turn"
    );
    assert!(play_year_of_plenty(&mut state, 0, 3, 4));
    assert_eq!(state.resources[0][3], 11);
    assert_eq!(state.resources[0][4], 11);
    assert_eq!(state.bank[3], 0);
}

#[test]
fn largest_army_awarded_at_three_knights_and_stolen_by_more() {
    let mut state = rich_state();
    state.dev_cards[0][DEV_KNIGHT] = 3;
    state.dev_cards[1][DEV_KNIGHT] = 4;

    for _ in 0..3 {
        assert!(play_knight(&mut state, 0));
        state.dev_card_played_this_turn = false; // simulate new turn
    }
    assert_eq!(state.largest_army_player, 0);
    assert_eq!(state.largest_army_size, 3);
    assert_eq!(state.calculate_victory_points(0), 2, "largest army = 2 VP");

    // Player 1 ties at 3: no steal. Takes over at 4.
    state.current_player = 1;
    for _ in 0..3 {
        assert!(play_knight(&mut state, 1));
        state.dev_card_played_this_turn = false;
    }
    assert_eq!(
        state.largest_army_player, 0,
        "tie does not transfer the award"
    );
    assert!(play_knight(&mut state, 1));
    assert_eq!(state.largest_army_player, 1);
    assert_eq!(state.largest_army_size, 4);
}
