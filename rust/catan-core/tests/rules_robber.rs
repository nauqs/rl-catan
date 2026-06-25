//! Robber mechanics: discard-on-7, robber movement, stealing.

use catan_core::board::topology;
use catan_core::building::build_settlement;
use catan_core::robber::{
    discard_one, get_players_who_must_discard, get_stealable_players, get_valid_robber_placements,
    move_robber, steal_random_resource,
};
use catan_core::state::GameState;

#[test]
fn discard_required_over_seven_cards() {
    let mut state = GameState::new(4, 5);
    state.resources[0] = [8, 0, 0, 0, 0]; // 8 cards -> discard 4
    state.resources[1] = [3, 4, 0, 0, 0]; // 7 cards -> safe
    state.resources[2] = [3, 3, 3, 0, 0]; // 9 cards -> discard 4 (floor)
    let must = get_players_who_must_discard(&state);
    assert_eq!(must, vec![(0, 4), (2, 4)]);
}

#[test]
fn discard_validates_and_returns_to_bank() {
    let mut state = GameState::new(4, 5);
    state.resources[0] = [2, 2, 0, 0, 0];
    let bank_before = state.bank;
    assert!(
        !discard_one(&mut state, 0, 2),
        "cannot discard an unheld resource"
    );
    assert!(discard_one(&mut state, 0, 0));
    assert!(discard_one(&mut state, 0, 0));
    assert!(!discard_one(&mut state, 0, 0), "hand exhausted for wheat");
    assert_eq!(state.resources[0][0], 0);
    assert_eq!(state.resources[0][1], 2);
    assert_eq!(state.bank[0], bank_before[0] + 2);
}

#[test]
fn robber_must_move_to_a_different_tile() {
    let mut state = GameState::new(4, 5);
    let here = state.robber_tile as usize;
    assert!(!move_robber(&mut state, here));
    let valid = get_valid_robber_placements(&state);
    assert_eq!(valid.len(), 18);
    assert!(!valid.contains(&here));
    let target = valid[0];
    assert!(move_robber(&mut state, target));
    assert_eq!(state.robber_tile as usize, target);
}

#[test]
fn stealable_players_excludes_self_and_empty_hands() {
    let mut state = GameState::new(4, 5);
    state.current_player = 0;
    let tile = 9usize; // center tile
    let verts = topology().tile_vertices[tile];
    assert!(build_settlement(&mut state, 0, verts[0] as usize, true));
    assert!(build_settlement(&mut state, 1, verts[2] as usize, true));
    assert!(build_settlement(&mut state, 2, verts[4] as usize, true));
    state.resources[1] = [1, 0, 0, 0, 0];
    state.resources[2] = [0, 0, 0, 0, 0]; // empty hand: not stealable

    let mut victims = get_stealable_players(&state, tile);
    victims.sort();
    assert_eq!(victims, vec![1]);
}

#[test]
fn steal_transfers_exactly_one_resource() {
    let mut state = GameState::new(4, 5);
    state.current_player = 0;
    state.resources[1] = [0, 0, 2, 0, 0];
    let stolen = steal_random_resource(&mut state, 1);
    assert_eq!(stolen, 2, "only wood available, must steal wood");
    assert_eq!(state.resources[1][2], 1);
    assert_eq!(state.resources[0][2], 1);

    // Stealing from an empty hand returns -1 and moves nothing.
    state.resources[3] = [0; 5];
    assert_eq!(steal_random_resource(&mut state, 3), -1);
}
