//! Development cards: buying, playing, largest army.

use crate::resources::{can_afford, pay_cost};
use crate::state::{
    GameState, DEV_CARD_COST, DEV_DECK_SIZE, DEV_KNIGHT, DEV_MONOPOLY, DEV_ROAD_BUILDING,
    DEV_VICTORY_POINT, DEV_YEAR_OF_PLENTY,
};

pub fn can_buy_dev_card(state: &GameState, player: usize) -> bool {
    can_afford(state, player, &DEV_CARD_COST) && state.dev_deck_idx < DEV_DECK_SIZE
}

/// Buy a development card. Returns the card type, or -1 on failure.
pub fn buy_dev_card(state: &mut GameState, player: usize) -> i32 {
    if !can_buy_dev_card(state, player) {
        return -1;
    }
    pay_cost(state, player, &DEV_CARD_COST);
    let card_type = state.dev_deck[state.dev_deck_idx];
    state.dev_deck_idx += 1;
    state.dev_cards[player][card_type as usize] += 1;
    state.dev_cards_bought_this_turn[card_type as usize] += 1;
    card_type as i32
}

pub fn can_play_dev_card(state: &GameState, player: usize, card_type: usize) -> bool {
    if state.dev_card_played_this_turn {
        return false;
    }
    if card_type == DEV_VICTORY_POINT {
        return false;
    }
    // Cards bought this turn cannot be played yet.
    let available =
        state.dev_cards[player][card_type] - state.dev_cards_bought_this_turn[card_type];
    available > 0
}

/// Play a knight. Caller handles robber movement and stealing.
pub fn play_knight(state: &mut GameState, player: usize) -> bool {
    if !can_play_dev_card(state, player, DEV_KNIGHT) {
        return false;
    }
    state.dev_cards[player][DEV_KNIGHT] -= 1;
    state.knights_played[player] += 1;
    state.dev_card_played_this_turn = true;
    update_largest_army(state);
    true
}

/// Play road building. Caller handles placing the 2 free roads.
pub fn play_road_building(state: &mut GameState, player: usize) -> bool {
    if !can_play_dev_card(state, player, DEV_ROAD_BUILDING) {
        return false;
    }
    state.dev_cards[player][DEV_ROAD_BUILDING] -= 1;
    state.dev_card_played_this_turn = true;
    true
}

pub fn play_year_of_plenty(
    state: &mut GameState,
    player: usize,
    resource1: usize,
    resource2: usize,
) -> bool {
    if !can_play_dev_card(state, player, DEV_YEAR_OF_PLENTY) {
        return false;
    }
    let mut needed = [0i16; 5];
    needed[resource1] += 1;
    needed[resource2] += 1;
    if (0..5).any(|r| state.bank[r] < needed[r]) {
        return false;
    }
    state.dev_cards[player][DEV_YEAR_OF_PLENTY] -= 1;
    state.dev_card_played_this_turn = true;
    for r in 0..5 {
        state.bank[r] -= needed[r];
        state.resources[player][r] += needed[r];
    }
    true
}

/// Play monopoly. Returns total resources taken, or -1 on failure.
pub fn play_monopoly(state: &mut GameState, player: usize, resource_type: usize) -> i32 {
    if !can_play_dev_card(state, player, DEV_MONOPOLY) {
        return -1;
    }
    if resource_type > 4 {
        return -1;
    }
    state.dev_cards[player][DEV_MONOPOLY] -= 1;
    state.dev_card_played_this_turn = true;

    let mut total = 0i32;
    for other in 0..state.num_players {
        if other != player {
            let amount = state.resources[other][resource_type];
            state.resources[other][resource_type] = 0;
            state.resources[player][resource_type] += amount;
            total += amount as i32;
        }
    }
    total
}

fn update_largest_army(state: &mut GameState) {
    for player in 0..state.num_players {
        let knights = state.knights_played[player];
        if knights >= 3 && knights > state.largest_army_size {
            state.largest_army_size = knights;
            state.largest_army_player = player as i8;
        }
    }
}
