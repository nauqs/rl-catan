//! Resource distribution and bank management.

use rand::Rng;

use crate::board::{topology, RESOURCE_DESERT};
use crate::state::{GameState, MAX_PLAYERS, NUM_RESOURCES};

/// Distribute resources for a dice roll. Mirrors the Python engine exactly:
/// per-vertex gains are capped at the (pre-roll) bank level, and if a
/// resource's total demand exceeds the bank, nobody receives that resource.
pub fn distribute_resources(state: &mut GameState, dice_roll: u8) {
    if dice_roll == 7 {
        return;
    }
    let topo = topology();
    let mut gained = [[0i16; NUM_RESOURCES]; MAX_PLAYERS];

    for &tile in &state.tiles_by_number[dice_roll as usize] {
        if tile < 0 {
            break;
        }
        let tile = tile as usize;
        if tile == state.robber_tile as usize {
            continue;
        }
        let res = state.tile_resources[tile];
        if res == RESOURCE_DESERT {
            continue;
        }
        let res = res as usize;
        for &vertex in &topo.tile_vertices[tile] {
            let val = state.vertices[vertex as usize];
            if val < 0 {
                continue;
            }
            let player = (val % 4) as usize;
            let amount: i16 = if val >= 4 { 2 } else { 1 };
            let amount = amount.min(state.bank[res]);
            if amount > 0 {
                gained[player][res] += amount;
            }
        }
    }

    for res in 0..NUM_RESOURCES {
        let total: i16 = (0..state.num_players).map(|p| gained[p][res]).sum();
        if total > state.bank[res] {
            // Bank can't cover demand: per standard rules, nobody gets it.
            continue;
        }
        state.bank[res] -= total;
        for p in 0..state.num_players {
            state.resources[p][res] += gained[p][res];
        }
    }
}

pub fn roll_dice(state: &mut GameState) -> (u8, u8, u8) {
    let d1 = state.rng.gen_range(1..=6u8);
    let d2 = state.rng.gen_range(1..=6u8);
    (d1, d2, d1 + d2)
}

pub fn can_afford(state: &GameState, player: usize, cost: &[i16; 5]) -> bool {
    (0..5).all(|r| state.resources[player][r] >= cost[r])
}

pub fn pay_cost(state: &mut GameState, player: usize, cost: &[i16; 5]) -> bool {
    if !can_afford(state, player, cost) {
        return false;
    }
    for r in 0..5 {
        state.resources[player][r] -= cost[r];
        state.bank[r] += cost[r];
    }
    true
}
