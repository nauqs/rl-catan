//! Bank trading: 4:1 default, 3:1 any-port, 2:1 resource ports, bank limits.

use catan_core::board::topology;
use catan_core::building::build_settlement;
use catan_core::state::GameState;
use catan_core::trading::{get_bank_trade_rate, get_possible_bank_trades, trade_with_bank};

const WHEAT: usize = 0;
const SHEEP: usize = 1;

#[test]
fn default_rate_is_four_to_one() {
    let state = GameState::new(4, 7);
    for r in 0..5 {
        assert_eq!(get_bank_trade_rate(&state, 0, r), 4);
    }
}

/// Find a port slot carrying the given (shuffled) port type.
fn port_vertex_of_type(state: &GameState, port_type: i8) -> usize {
    let slot = (0..9).find(|&p| state.port_types[p] == port_type).unwrap();
    topology().port_vertices[slot][0] as usize
}

#[test]
fn any_port_gives_three_to_one() {
    let mut state = GameState::new(4, 7);
    let v = port_vertex_of_type(&state, 0); // 3:1 "any" port
    assert!(build_settlement(&mut state, 0, v, true));
    for r in 0..5 {
        assert_eq!(get_bank_trade_rate(&state, 0, r), 3);
    }
    // Other players unaffected.
    assert_eq!(get_bank_trade_rate(&state, 1, 0), 4);
}

#[test]
fn resource_port_gives_two_to_one_for_that_resource_only() {
    let mut state = GameState::new(4, 7);
    let v = port_vertex_of_type(&state, WHEAT as i8 + 1); // 2:1 wheat port
    assert!(build_settlement(&mut state, 0, v, true));
    assert_eq!(get_bank_trade_rate(&state, 0, WHEAT), 2);
    for r in 1..5 {
        assert_eq!(
            get_bank_trade_rate(&state, 0, r),
            4,
            "resource {r} stays 4:1"
        );
    }
}

#[test]
fn trade_with_bank_moves_resources_at_rate() {
    let mut state = GameState::new(4, 7);
    state.resources[0] = [4, 0, 0, 0, 0];
    let bank_before = state.bank;
    assert!(trade_with_bank(&mut state, 0, WHEAT, SHEEP));
    assert_eq!(state.resources[0][WHEAT], 0);
    assert_eq!(state.resources[0][SHEEP], 1);
    assert_eq!(state.bank[WHEAT], bank_before[WHEAT] + 4);
    assert_eq!(state.bank[SHEEP], bank_before[SHEEP] - 1);
}

#[test]
fn invalid_trades_are_rejected() {
    let mut state = GameState::new(4, 7);
    state.resources[0] = [4, 0, 0, 0, 0];
    // Same resource both ways.
    assert!(!trade_with_bank(&mut state, 0, WHEAT, WHEAT));
    // Not enough to give.
    assert!(!trade_with_bank(&mut state, 0, SHEEP, WHEAT));
    // Bank out of the requested resource.
    state.bank[SHEEP] = 0;
    assert!(!trade_with_bank(&mut state, 0, WHEAT, SHEEP));
    assert_eq!(
        state.resources[0][WHEAT], 4,
        "failed trades must not move resources"
    );
}

#[test]
fn possible_trades_enumeration_matches_holdings() {
    let mut state = GameState::new(4, 7);
    state.resources[0] = [4, 3, 0, 0, 0];
    let trades = get_possible_bank_trades(&state, 0);
    // Wheat at 4:1 -> 4 target resources. Sheep (only 3, rate 4) -> none.
    assert_eq!(trades.len(), 4);
    for &(give, amount, recv) in &trades {
        assert_eq!(give, WHEAT);
        assert_eq!(amount, 4);
        assert_ne!(recv, WHEAT);
    }
}
