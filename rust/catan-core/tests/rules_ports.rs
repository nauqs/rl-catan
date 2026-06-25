//! Ports: the per-game port-type shuffle must actually govern trade rates.

use catan_core::board::topology;
use catan_core::building::build_settlement;
use catan_core::state::GameState;
use catan_core::trading::get_bank_trade_rate;

/// Find a port slot whose shuffled type matches, returning one of its vertices.
fn vertex_of_port_type(state: &GameState, port_type: i8) -> Option<usize> {
    let topo = topology();
    (0..9)
        .find(|&p| state.port_types[p] == port_type)
        .map(|p| topo.port_vertices[p][0] as usize)
}

#[test]
fn shuffled_port_types_govern_trade_rates() {
    let mut state = GameState::new(4, 7);
    // Wherever the shuffle put the 2:1 wheat port, settling there gives 2:1
    // wheat (port type wheat+1 == 1).
    let v = vertex_of_port_type(&state, 1).expect("board has a wheat port");
    assert!(build_settlement(&mut state, 0, v, true));
    assert_eq!(
        get_bank_trade_rate(&state, 0, 0),
        2,
        "wheat at the wheat port"
    );
    for r in 1..5 {
        assert_eq!(
            get_bank_trade_rate(&state, 0, r),
            4,
            "other resources stay 4:1"
        );
    }
}

#[test]
fn any_port_gives_three_to_one_wherever_it_lands() {
    let mut state = GameState::new(4, 7);
    let v = vertex_of_port_type(&state, 0).expect("board has a 3:1 port");
    assert!(build_settlement(&mut state, 1, v, true));
    for r in 0..5 {
        assert_eq!(get_bank_trade_rate(&state, 1, r), 3);
    }
}

#[test]
fn port_layouts_differ_across_seeds() {
    // The shuffle must produce different port layouts for different games.
    let layouts: Vec<[i8; 9]> = (0..20u64)
        .map(|s| GameState::new(4, s).port_types)
        .collect();
    assert!(
        layouts.windows(2).any(|w| w[0] != w[1]),
        "port types never vary across seeds — shuffle not applied"
    );
}

#[test]
fn non_port_vertices_trade_at_four() {
    let state = GameState::new(4, 7);
    let topo = topology();
    // A vertex with no port index must contribute nothing.
    let v = (0..54).find(|&v| topo.vertex_port_index[v] < 0).unwrap();
    let mut state2 = state.clone();
    assert!(build_settlement(&mut state2, 2, v, true));
    for r in 0..5 {
        assert_eq!(get_bank_trade_rate(&state2, 2, r), 4);
    }
}
