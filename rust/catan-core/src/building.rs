//! Building: placement validation, construction, longest road.

use crate::board::topology;
use crate::resources::{can_afford, pay_cost};
use crate::state::{GameState, CITY_COST, ROAD_COST, SETTLEMENT_COST};

pub fn can_afford_road(state: &GameState, player: usize) -> bool {
    can_afford(state, player, &ROAD_COST)
}

pub fn can_afford_settlement(state: &GameState, player: usize) -> bool {
    can_afford(state, player, &SETTLEMENT_COST)
}

pub fn can_afford_city(state: &GameState, player: usize) -> bool {
    can_afford(state, player, &CITY_COST)
}

/// Is this empty edge connected to the player's network (own building on an
/// endpoint, or own road through an endpoint not blocked by an opponent)?
/// O(1): reads the per-player road-touch bitmask maintained by `build_road`.
fn road_edge_connected(state: &GameState, player: usize, edge_idx: usize) -> bool {
    let player_i8 = player as i8;
    let road_mask = state.vertex_road_mask[player];
    let [v1, v2] = topology().edge_vertices[edge_idx];
    for &v in &[v1, v2] {
        if state.occupied_mask & (1u64 << v) == 0 {
            // Empty vertex: connected iff one of our roads touches it.
            if road_mask & (1u64 << v) != 0 {
                return true;
            }
        } else {
            let owner = state.settlement_owner(v as usize);
            if owner == player_i8 {
                return true;
            }
            // Opponent building: blocks continuation through this vertex.
        }
    }
    false
}

/// Full validity of a single road placement (used by `execute_action`).
pub fn road_placement_ok(state: &GameState, player: usize, edge_idx: usize, free: bool) -> bool {
    edge_idx < 72
        && state.edges[edge_idx] < 0
        && (free || can_afford_road(state, player))
        && state.roads_built[player] < state.max_roads
        && road_edge_connected(state, player, edge_idx)
}

/// Does the player have at least one legal free road placement?
pub fn has_free_road_placement(state: &GameState, player: usize) -> bool {
    if state.roads_built[player] >= state.max_roads {
        return false;
    }
    (0..72).any(|e| state.edges[e] < 0 && road_edge_connected(state, player, e))
}

/// Visit every edge where the player may build a road. Allocation-free; the
/// `get_valid_*` wrappers collect into a Vec for convenience.
pub fn for_each_valid_road_placement(
    state: &GameState,
    player: usize,
    free_placement: bool,
    mut f: impl FnMut(usize),
) {
    if !free_placement && !can_afford_road(state, player) {
        return;
    }
    if state.roads_built[player] >= state.max_roads {
        return;
    }
    for edge_idx in 0..72 {
        if state.edges[edge_idx] < 0 && road_edge_connected(state, player, edge_idx) {
            f(edge_idx);
        }
    }
}

pub fn get_valid_road_placements(
    state: &GameState,
    player: usize,
    free_placement: bool,
) -> Vec<usize> {
    let mut valid = Vec::new();
    for_each_valid_road_placement(state, player, free_placement, |e| valid.push(e));
    valid
}

/// Full validity of a single settlement placement (used by `execute_action`).
pub fn settlement_placement_ok(
    state: &GameState,
    player: usize,
    vertex_idx: usize,
    setup_phase: bool,
) -> bool {
    if vertex_idx >= 54 {
        return false;
    }
    let bit = 1u64 << vertex_idx;
    // Vertex empty + distance rule, both as single mask tests.
    if state.occupied_mask & bit != 0
        || state.occupied_mask & topology().neighbor_mask[vertex_idx] != 0
    {
        return false;
    }
    if state.settlements_built[player] >= state.max_settlements {
        return false;
    }
    if !setup_phase {
        if state.vertex_road_mask[player] & bit == 0 {
            return false;
        }
        if !can_afford_settlement(state, player) {
            return false;
        }
    }
    true
}

/// Full validity of a single city placement (used by `execute_action`).
pub fn city_placement_ok(state: &GameState, player: usize, vertex_idx: usize) -> bool {
    vertex_idx < 54
        && state.vertices[vertex_idx] == player as i8
        && state.cities_built[player] < state.max_cities
        && can_afford_city(state, player)
}

/// Length of the player's longest road (public for oracle cross-checks).
pub fn longest_road_length(state: &GameState, player: usize) -> usize {
    calculate_longest_road_for_player(state, player)
}

/// Visit every vertex where the player may build a settlement.
pub fn for_each_valid_settlement_placement(
    state: &GameState,
    player: usize,
    setup_phase: bool,
    mut f: impl FnMut(usize),
) {
    if !setup_phase && !can_afford_settlement(state, player) {
        return;
    }
    if state.settlements_built[player] >= state.max_settlements {
        return;
    }

    let topo = topology();
    // In the main phase only road-touched vertices are candidates; in setup
    // every vertex is.
    let candidates = if setup_phase {
        (1u128 << 54) - 1
    } else {
        state.vertex_road_mask[player] as u128
    };
    let mut remaining = candidates;
    while remaining != 0 {
        let vertex_idx = remaining.trailing_zeros() as usize;
        remaining &= remaining - 1;
        let bit = 1u64 << vertex_idx;
        // Vertex empty + distance rule, both as single mask tests.
        if state.occupied_mask & bit == 0
            && state.occupied_mask & topo.neighbor_mask[vertex_idx] == 0
        {
            f(vertex_idx);
        }
    }
}

pub fn get_valid_settlement_placements(
    state: &GameState,
    player: usize,
    setup_phase: bool,
) -> Vec<usize> {
    let mut valid = Vec::new();
    for_each_valid_settlement_placement(state, player, setup_phase, |v| valid.push(v));
    valid
}

/// Visit every vertex where the player may upgrade to a city.
pub fn for_each_valid_city_placement(state: &GameState, player: usize, mut f: impl FnMut(usize)) {
    if !can_afford_city(state, player) {
        return;
    }
    if state.cities_built[player] >= state.max_cities {
        return;
    }
    for v in 0..54 {
        if state.vertices[v] == player as i8 {
            f(v);
        }
    }
}

pub fn get_valid_city_placements(state: &GameState, player: usize) -> Vec<usize> {
    let mut valid = Vec::new();
    for_each_valid_city_placement(state, player, |v| valid.push(v));
    valid
}

pub fn build_road(state: &mut GameState, player: usize, edge_idx: usize, free: bool) -> bool {
    if edge_idx >= 72 || state.edges[edge_idx] >= 0 {
        return false;
    }
    if !free && !pay_cost(state, player, &ROAD_COST) {
        return false;
    }
    state.edges[edge_idx] = player as i8;
    state.roads_built[player] += 1;
    let [v1, v2] = topology().edge_vertices[edge_idx];
    state.vertex_road_mask[player] |= (1u64 << v1) | (1u64 << v2);
    // A new road can only lengthen the builder's own longest road: other
    // players' road graphs are untouched (blocking depends on settlements).
    state.road_lengths[player] = calculate_longest_road_for_player(state, player) as u8;
    apply_longest_road_award(state);
    true
}

pub fn build_settlement(
    state: &mut GameState,
    player: usize,
    vertex_idx: usize,
    free: bool,
) -> bool {
    if vertex_idx >= 54 || state.vertices[vertex_idx] >= 0 {
        return false;
    }
    for &n in &topology().vertex_neighbors[vertex_idx] {
        if n >= 0 && state.vertices[n as usize] >= 0 {
            return false;
        }
    }
    if !free && !pay_cost(state, player, &SETTLEMENT_COST) {
        return false;
    }
    state.vertices[vertex_idx] = player as i8;
    state.settlements_built[player] += 1;
    state.occupied_mask |= 1u64 << vertex_idx;

    // Keep the port-rate cache current (ports are never lost once gained).
    let port_type = state.vertex_port_type(vertex_idx);
    if port_type == 0 {
        state.port_any[player] = true;
    } else if port_type > 0 {
        state.port_resource[player][(port_type - 1) as usize] = true;
    }

    // A new settlement can only break roads passing THROUGH this vertex:
    // recompute exactly the players with a road touching it.
    let bit = 1u64 << vertex_idx;
    let mut any_affected = false;
    for p in 0..state.num_players {
        if state.vertex_road_mask[p] & bit != 0 {
            state.road_lengths[p] = calculate_longest_road_for_player(state, p) as u8;
            any_affected = true;
        }
    }
    if any_affected {
        apply_longest_road_award(state);
    }
    true
}

pub fn build_city(state: &mut GameState, player: usize, vertex_idx: usize) -> bool {
    if vertex_idx >= 54 || state.vertices[vertex_idx] != player as i8 {
        return false;
    }
    if !pay_cost(state, player, &CITY_COST) {
        return false;
    }
    state.vertices[vertex_idx] = player as i8 + 4;
    state.settlements_built[player] -= 1;
    state.cities_built[player] += 1;
    true
}

fn calculate_longest_road_for_player(state: &GameState, player: usize) -> usize {
    let player_i8 = player as i8;

    let mut edge_mask: u128 = 0;
    let mut player_edges = [0u8; 15];
    let mut num_edges = 0usize;
    for e in 0..72u8 {
        if state.edges[e as usize] == player_i8 {
            edge_mask |= 1u128 << e;
            player_edges[num_edges] = e;
            num_edges += 1;
        }
    }
    // Below 5 edges the exact trail length never matters for the award:
    // return the raw count (always >= the true trail length, still < 5).
    if num_edges < 5 {
        return num_edges;
    }

    // Directed trail walk: arriving at a vertex, continue only through it
    // (blocked by opponents' buildings), never reusing an edge. Roads are
    // physical segments — a "path" cannot pivot back through the vertex it
    // just crossed, which is why the walk tracks the exit vertex.
    fn walk(at: u8, used: u128, edge_mask: u128, state: &GameState, player_i8: i8) -> usize {
        let owner = state.settlement_owner(at as usize);
        if owner >= 0 && owner != player_i8 {
            return 0; // opponent settlement blocks continuation
        }
        let topo = topology();
        let mut best = 0;
        for &adj in &topo.vertex_edges[at as usize] {
            if adj >= 0 {
                let e = adj as u8;
                let bit = 1u128 << e;
                if edge_mask & bit != 0 && used & bit == 0 {
                    let [a, b] = topo.edge_vertices[e as usize];
                    let next = if a == at { b } else { a };
                    best = best.max(1 + walk(next, used | bit, edge_mask, state, player_i8));
                }
            }
        }
        best
    }

    let topo = topology();
    let mut longest = 0;
    for &start in &player_edges[..num_edges] {
        let [a, b] = topo.edge_vertices[start as usize];
        let bit = 1u128 << start;
        longest = longest.max(1 + walk(b, bit, edge_mask, state, player_i8));
        longest = longest.max(1 + walk(a, bit, edge_mask, state, player_i8));
    }
    longest
}

/// Apply the official award rules from the cached per-player lengths:
/// - the card needs a road of at least 5;
/// - the current holder keeps it on ties;
/// - a tie among non-holders sets the card aside (nobody scores it).
fn apply_longest_road_award(state: &mut GameState) {
    let lengths = &state.road_lengths[..state.num_players];
    let max_len = lengths.iter().copied().max().unwrap_or(0);
    if max_len < 5 {
        state.longest_road_player = -1;
        state.longest_road_length = 0;
        return;
    }
    let holder = state.longest_road_player;
    if holder >= 0 && lengths[holder as usize] == max_len {
        state.longest_road_length = max_len;
        return; // holder retains on ties
    }
    let mut leader: i8 = -1;
    let mut tie = false;
    for (p, &len) in lengths.iter().enumerate() {
        if len == max_len {
            tie = leader >= 0;
            leader = p as i8;
        }
    }
    if tie {
        // Set aside until someone has the sole longest road.
        state.longest_road_player = -1;
        state.longest_road_length = 0;
    } else {
        state.longest_road_player = leader;
        state.longest_road_length = max_len;
    }
}
