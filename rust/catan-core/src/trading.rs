//! Bank trading: 4:1 default, 3:1 any-port, 2:1 resource ports.
//!
//! Rates come from the per-player port cache on `GameState`, which
//! `build_settlement` keeps current — no vertex scans in the hot path.

use crate::state::GameState;

pub fn get_bank_trade_rate(state: &GameState, player: usize, resource_type: usize) -> i16 {
    if state.port_resource[player][resource_type] {
        2
    } else if state.port_any[player] {
        3
    } else {
        4
    }
}

pub fn can_trade_with_bank(
    state: &GameState,
    player: usize,
    give_resource: usize,
    give_amount: i16,
    receive_resource: usize,
) -> bool {
    if give_resource == receive_resource {
        return false;
    }
    let rate = get_bank_trade_rate(state, player, give_resource);
    if give_amount != rate {
        return false;
    }
    if state.resources[player][give_resource] < give_amount {
        return false;
    }
    if state.bank[receive_resource] < 1 {
        return false;
    }
    true
}

pub fn trade_with_bank(
    state: &mut GameState,
    player: usize,
    give_resource: usize,
    receive_resource: usize,
) -> bool {
    let rate = get_bank_trade_rate(state, player, give_resource);
    if !can_trade_with_bank(state, player, give_resource, rate, receive_resource) {
        return false;
    }
    state.resources[player][give_resource] -= rate;
    state.bank[give_resource] += rate;
    state.resources[player][receive_resource] += 1;
    state.bank[receive_resource] -= 1;
    true
}

/// Visit every possible bank trade as (give_resource, give_amount,
/// receive_resource), in the same order the Python engine enumerates them.
pub fn for_each_bank_trade(state: &GameState, player: usize, mut f: impl FnMut(usize, i16, usize)) {
    for give in 0..5 {
        let rate = get_bank_trade_rate(state, player, give);
        if state.resources[player][give] >= rate {
            for recv in 0..5 {
                if recv != give && state.bank[recv] > 0 {
                    f(give, rate, recv);
                }
            }
        }
    }
}

pub fn get_possible_bank_trades(state: &GameState, player: usize) -> Vec<(usize, i16, usize)> {
    let mut trades = Vec::new();
    for_each_bank_trade(state, player, |g, a, r| trades.push((g, a, r)));
    trades
}
