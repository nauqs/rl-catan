//! Building rules: placement validation, costs, distance rule, longest road.

use catan_core::board::topology;
use catan_core::building::{
    build_city, build_road, build_settlement, get_valid_road_placements,
    get_valid_settlement_placements,
};
use catan_core::state::{GameState, CITY_COST, ROAD_COST, SETTLEMENT_COST};

fn fresh_state() -> GameState {
    GameState::new(4, 42)
}

/// Find the edge index connecting two vertices.
fn edge_between(v1: u8, v2: u8) -> usize {
    let topo = topology();
    let key = [v1.min(v2), v1.max(v2)];
    (0..72)
        .find(|&e| topo.edge_vertices[e] == key)
        .expect("edge exists")
}

#[test]
fn settlement_distance_rule_blocks_neighbors() {
    let mut state = fresh_state();
    assert!(build_settlement(&mut state, 0, 0, true));
    for &n in topology().vertex_neighbors[0].iter().filter(|&&n| n >= 0) {
        assert!(
            !build_settlement(&mut state, 1, n as usize, true),
            "vertex {n} adjacent to occupied vertex 0 must be rejected"
        );
    }
    // Setup-phase valid placements must exclude vertex 0 and its neighbors.
    let valid = get_valid_settlement_placements(&state, 1, true);
    assert!(!valid.contains(&0));
    for &n in topology().vertex_neighbors[0].iter().filter(|&&n| n >= 0) {
        assert!(!valid.contains(&(n as usize)));
    }
}

#[test]
fn settlement_requires_road_connectivity_outside_setup() {
    let mut state = fresh_state();
    state.resources[0] = [10, 10, 10, 10, 10];
    // No roads built: no valid main-phase settlement spots.
    assert!(get_valid_settlement_placements(&state, 0, false).is_empty());
    // After placing a settlement + adjacent roads, the far end of a 2-road
    // chain becomes valid.
    assert!(build_settlement(&mut state, 0, 0, true));
    let topo = topology();
    let n1 = topo.vertex_neighbors[0][0] as u8;
    let e1 = edge_between(0, n1);
    assert!(build_road(&mut state, 0, e1, true));
    let n2 = topo.vertex_neighbors[n1 as usize]
        .iter()
        .copied()
        .find(|&v| v > 0)
        .unwrap() as u8;
    let e2 = edge_between(n1, n2);
    assert!(build_road(&mut state, 0, e2, true));
    let valid = get_valid_settlement_placements(&state, 0, false);
    assert!(
        valid.contains(&(n2 as usize)),
        "end of road chain should be buildable"
    );
}

#[test]
fn building_costs_are_charged_to_player_and_credited_to_bank() {
    let mut state = fresh_state();
    state.resources[0] = [5, 5, 5, 5, 5];
    let bank_before = state.bank;

    assert!(build_settlement(&mut state, 0, 10, false));
    for r in 0..5 {
        assert_eq!(state.resources[0][r], 5 - SETTLEMENT_COST[r]);
        assert_eq!(state.bank[r], bank_before[r] + SETTLEMENT_COST[r]);
    }

    let res_before = state.resources[0];
    let topo = topology();
    let n = topo.vertex_neighbors[10][0] as u8;
    let e = edge_between(10, n);
    assert!(build_road(&mut state, 0, e, false));
    for r in 0..5 {
        assert_eq!(state.resources[0][r], res_before[r] - ROAD_COST[r]);
    }

    let res_before = state.resources[0];
    assert!(build_city(&mut state, 0, 10));
    for r in 0..5 {
        assert_eq!(state.resources[0][r], res_before[r] - CITY_COST[r]);
    }
}

#[test]
fn cannot_build_without_resources() {
    let mut state = fresh_state();
    state.resources[0] = [0, 0, 0, 0, 0];
    assert!(!build_settlement(&mut state, 0, 10, false));
    assert!(!build_road(&mut state, 0, 0, false));
    assert!(get_valid_road_placements(&state, 0, false).is_empty());
}

#[test]
fn city_upgrade_replaces_settlement() {
    let mut state = fresh_state();
    state.resources[0] = [2, 0, 0, 0, 3];
    assert!(build_settlement(&mut state, 0, 20, true));
    assert_eq!(state.settlements_built[0], 1);

    // Cannot upgrade someone else's settlement or an empty vertex.
    state.resources[1] = [2, 0, 0, 0, 3];
    assert!(!build_city(&mut state, 1, 20));
    assert!(!build_city(&mut state, 0, 21));

    assert!(build_city(&mut state, 0, 20));
    assert_eq!(state.vertices[20], 4); // player 0 city encoding
    assert_eq!(state.settlements_built[0], 0);
    assert_eq!(state.cities_built[0], 1);
    assert_eq!(state.settlement_owner(20), 0);
    assert!(state.is_city(20));
    // City counts 2 VP.
    assert_eq!(state.calculate_victory_points(0), 2);
}

#[test]
fn road_placements_require_connection_to_own_network() {
    let mut state = fresh_state();
    state.resources[0] = [10, 10, 10, 10, 10];
    assert!(build_settlement(&mut state, 0, 0, true));
    let valid = get_valid_road_placements(&state, 0, false);
    // Exactly the empty edges touching vertex 0 are valid right now.
    let topo = topology();
    let expected: Vec<usize> = topo.vertex_edges[0]
        .iter()
        .filter(|&&e| e >= 0)
        .map(|&e| e as usize)
        .collect();
    assert_eq!(valid, expected);
}

#[test]
fn longest_road_requires_five_and_tracks_holder() {
    let mut state = fresh_state();
    // Build around tile 0's hex ring: 6 vertices -> 6 edges in a cycle.
    let ring = topology().tile_vertices[0];
    for i in 0..4 {
        let e = edge_between(ring[i], ring[(i + 1) % 6]);
        assert!(build_road(&mut state, 0, e, true));
    }
    assert_eq!(state.longest_road_player, -1, "4 roads is not enough");

    let e = edge_between(ring[4], ring[5]);
    assert!(build_road(&mut state, 0, e, true));
    // 5 roads but split: 0-1-2-3-4 chain + 4-5 edge connect (4..5 adjacent) -> still one chain of 5.
    assert_eq!(state.longest_road_player, 0);
    assert!(state.longest_road_length >= 5);
    assert_eq!(state.calculate_victory_points(0), 2, "longest road = 2 VP");
}

#[test]
fn opponent_settlement_breaks_longest_road() {
    let mut state = fresh_state();
    let ring = topology().tile_vertices[0];
    for i in 0..5 {
        let e = edge_between(ring[i], ring[(i + 1) % 6]);
        assert!(build_road(&mut state, 0, e, true));
    }
    assert_eq!(state.longest_road_player, 0);

    // Opponent settlement in the middle of the chain splits it 2/3.
    assert!(build_settlement(&mut state, 1, ring[2] as usize, true));
    assert_eq!(state.longest_road_player, -1, "broken road loses the award");
}
