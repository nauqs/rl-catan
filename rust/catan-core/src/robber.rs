//! Robber: discard-on-7, movement, stealing.

use rand::Rng;

use crate::board::topology;
use crate::state::GameState;

/// Players holding more than 7 cards must discard half (rounded down).
/// Fills `out` (cleared first) so the hot path reuses its buffer.
pub fn get_players_who_must_discard_into(state: &GameState, out: &mut Vec<(usize, u8)>) {
    out.clear();
    for p in 0..state.num_players {
        let total = state.total_resources(p);
        if total > 7 {
            out.push((p, (total / 2) as u8));
        }
    }
}

pub fn get_players_who_must_discard(state: &GameState) -> Vec<(usize, u8)> {
    let mut out = Vec::new();
    get_players_who_must_discard_into(state, &mut out);
    out
}

/// Discard a single card to the bank (the 7-roll flow discards one at a
/// time, chosen by the discarding player).
pub fn discard_one(state: &mut GameState, player: usize, resource: usize) -> bool {
    if resource >= 5 || state.resources[player][resource] < 1 {
        return false;
    }
    state.resources[player][resource] -= 1;
    state.bank[resource] += 1;
    true
}

pub fn get_valid_robber_placements(state: &GameState) -> Vec<usize> {
    (0..19)
        .filter(|&t| t != state.robber_tile as usize)
        .collect()
}

/// Allocation-free form of `get_stealable_players`: per-player flags for
/// "has a building on this tile, isn't the current player, holds resources".
pub fn stealable_flags(state: &GameState, tile: usize) -> [bool; 4] {
    let mut found = [false; 4];
    for &vertex in &topology().tile_vertices[tile] {
        let owner = state.settlement_owner(vertex as usize);
        if owner >= 0
            && owner != state.current_player as i8
            && state.total_resources(owner as usize) > 0
        {
            found[owner as usize] = true;
        }
    }
    found
}

/// Players with a building on the tile, excluding the current player and
/// players with no resources. Sorted by player index.
pub fn get_stealable_players(state: &GameState, tile: usize) -> Vec<usize> {
    let found = stealable_flags(state, tile);
    (0..state.num_players).filter(|&p| found[p]).collect()
}

pub fn move_robber(state: &mut GameState, tile: usize) -> bool {
    if tile >= 19 || tile == state.robber_tile as usize {
        return false;
    }
    state.robber_tile = tile as u8;
    true
}

/// Steal one random resource from a player. Returns the resource type, or -1
/// if the victim has nothing.
pub fn steal_random_resource(state: &mut GameState, victim: usize) -> i8 {
    let total = state.total_resources(victim);
    if total == 0 {
        return -1;
    }
    let pick = state.rng.gen_range(0..total);
    let mut cumulative = 0i16;
    for resource_type in 0..5 {
        cumulative += state.resources[victim][resource_type];
        if pick < cumulative {
            return steal_specific_resource(state, victim, resource_type as i8);
        }
    }
    -1
}

/// Steal a specific resource type (used by replay with recorded outcomes).
pub fn steal_specific_resource(state: &mut GameState, victim: usize, resource_type: i8) -> i8 {
    if resource_type < 0 {
        return -1;
    }
    let r = resource_type as usize;
    let thief = state.current_player;
    state.resources[victim][r] -= 1;
    state.resources[thief][r] += 1;
    resource_type
}
