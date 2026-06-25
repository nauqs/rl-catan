//! Observation encoder: a fixed-length f32 vector view of the game from one
//! seat's perspective — the policy network's input.
//!
//! Design rules (mirrors of the codec's):
//! - Fixed layout, frozen once training starts: slot i always means the
//!   same thing. `OBS_VERSION` is stored in checkpoints and must match.
//! - Seat-relative: players appear in the order me, +1, +2, +3 around the
//!   table, so one network generalizes across seats.
//! - Categorical data is one-hot (a network must not read resource id 3 as
//!   "more than" id 2); scalars are normalized to roughly [0, 1]; dice
//!   numbers are encoded as production PROBABILITY, not face value.
//! - Two visibility modes with the same shape: `Perfect` shows opponents'
//!   hands and dev cards; `Realistic` zeroes exactly that block (counts
//!   stay visible — they're public in real Catan).
//!
//! Layout (start offset, count):
//!   tiles        19 x 8   [resource one-hot(6) | production prob | robber]
//!   vertices     54 x 14  [building by rel seat: (settle, city) x4 | port one-hot(6)]
//!   edges        72 x 4   [road owner rel one-hot]
//!   players       4 x 17  public per rel seat (see PLAYER_* consts)
//!   self private      15  [hand(5) | dev by type(5) | bought this turn(5)]
//!   opp private   3 x 10  [hand(5) | dev by type(5)] — zeroed in Realistic
//!   bank               5
//!   context           20  [game phase(4) | turn phase(9) | dice | rolled |
//!                          turn | victory target | roads to place |
//!                          trades left | my discards due]
//!   trade offer       16  [present | proposer rel(4) | give(5) | amount | recv(5)]

use catan_core::board::topology;
use catan_core::game::{CatanGame, GamePhase, TurnPhase};
use catan_core::state::DEV_VICTORY_POINT;

pub const OBS_VERSION: u32 = 1;

pub const TILES: usize = 0;
pub const TILE_STRIDE: usize = 8;
pub const VERTICES: usize = TILES + 19 * TILE_STRIDE; // 152
pub const VERTEX_STRIDE: usize = 14;
pub const EDGES: usize = VERTICES + 54 * VERTEX_STRIDE; // 908
pub const EDGE_STRIDE: usize = 4;
pub const PLAYERS: usize = EDGES + 72 * EDGE_STRIDE; // 1196
pub const PLAYER_STRIDE: usize = 17;
pub const SELF_PRIVATE: usize = PLAYERS + 4 * PLAYER_STRIDE; // 1264
pub const OPP_PRIVATE: usize = SELF_PRIVATE + 15; // 1279
pub const OPP_STRIDE: usize = 10;
pub const BANK: usize = OPP_PRIVATE + 3 * OPP_STRIDE; // 1309
pub const CONTEXT: usize = BANK + 5; // 1314
pub const TRADE: usize = CONTEXT + 20; // 1334
pub const OBS_DIM: usize = TRADE + 16; // 1350

// Player-block slot indices (within PLAYER_STRIDE).
pub const PLAYER_CARDS: usize = 0; // total cards /19 (public)
pub const PLAYER_DEV_COUNT: usize = 1; // dev cards held /25 (public)
pub const PLAYER_KNIGHTS: usize = 2; // knights played /14
pub const PLAYER_PUBLIC_VP: usize = 3; // VP excl. hidden VP cards / target
pub const PLAYER_SETTLEMENTS_LEFT: usize = 4;
pub const PLAYER_CITIES_LEFT: usize = 5;
pub const PLAYER_ROADS_LEFT: usize = 6;
pub const PLAYER_ROAD_LEN: usize = 7;
pub const PLAYER_HAS_LONGEST: usize = 8;
pub const PLAYER_HAS_LARGEST: usize = 9;
pub const PLAYER_PORT_ANY: usize = 10;
pub const PLAYER_PORT_2TO1: usize = 11; // ..=15, one per resource
pub const PLAYER_IS_TURN_OWNER: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// God view: opponents' hands and dev cards visible. Easier learning;
    /// the first training target.
    Perfect,
    /// Real Catan: opponents' private block zeroed (their card COUNTS stay
    /// visible — those are public at a real table).
    Realistic,
}

fn rel_of(seat: usize, other: usize, n: usize) -> usize {
    (other + n - seat) % n
}

/// Encode the game as seen from `seat` into `out` (length OBS_DIM).
/// Allocation-free; zeroes the buffer first.
pub fn encode_obs(game: &CatanGame, seat: usize, visibility: Visibility, out: &mut [f32]) {
    assert_eq!(out.len(), OBS_DIM, "obs buffer must be OBS_DIM");
    out.fill(0.0);
    let s = &game.state;
    let n = s.num_players;
    let topo = topology();
    assert!(seat < n, "seat {seat} out of range");

    // ---- tiles: resource one-hot, production probability, robber --------
    for t in 0..19 {
        let base = TILES + t * TILE_STRIDE;
        out[base + s.tile_resources[t] as usize] = 1.0;
        let number = s.tile_numbers[t];
        if number > 0 {
            // number_probabilities peaks at 5/36 (6s and 8s); x7.2 -> 1.0.
            out[base + 6] = topo.number_probabilities[number as usize] * 7.2;
        }
        if t as u8 == s.robber_tile {
            out[base + 7] = 1.0;
        }
    }

    // ---- vertices: building by relative owner + static port -------------
    for v in 0..54 {
        let base = VERTICES + v * VERTEX_STRIDE;
        let val = s.vertices[v];
        if val >= 0 {
            let owner = (val % 4) as usize;
            let is_city = val >= 4;
            let rel = rel_of(seat, owner, n);
            out[base + rel * 2 + usize::from(is_city)] = 1.0;
        }
        let port = s.vertex_port_type(v);
        if port >= 0 {
            out[base + 8 + port as usize] = 1.0; // 0 = any, 1..=5 = 2:1
        }
    }

    // ---- edges: road owner, relative ------------------------------------
    for e in 0..72 {
        let owner = s.edges[e];
        if owner >= 0 {
            let rel = rel_of(seat, owner as usize, n);
            out[EDGES + e * EDGE_STRIDE + rel] = 1.0;
        }
    }

    // ---- players: public info per relative seat -------------------------
    let target = s.victory_target as f32;
    for rel in 0..n {
        let p = (seat + rel) % n;
        let base = PLAYERS + rel * PLAYER_STRIDE;
        out[base + PLAYER_CARDS] = s.total_resources(p) as f32 / 19.0;
        let dev_count: i32 = s.dev_cards[p].iter().map(|&c| c as i32).sum();
        out[base + PLAYER_DEV_COUNT] = dev_count as f32 / 25.0;
        out[base + PLAYER_KNIGHTS] = s.knights_played[p] as f32 / 14.0;
        let public_vp = s.calculate_victory_points(p) - s.dev_cards[p][DEV_VICTORY_POINT] as i32;
        out[base + PLAYER_PUBLIC_VP] = public_vp as f32 / target;
        out[base + PLAYER_SETTLEMENTS_LEFT] =
            (s.max_settlements - s.settlements_built[p]) as f32 / 5.0;
        out[base + PLAYER_CITIES_LEFT] = (s.max_cities - s.cities_built[p]) as f32 / 4.0;
        out[base + PLAYER_ROADS_LEFT] = (s.max_roads - s.roads_built[p]) as f32 / 15.0;
        out[base + PLAYER_ROAD_LEN] = s.road_lengths[p] as f32 / 15.0;
        out[base + PLAYER_HAS_LONGEST] = f32::from(s.longest_road_player == p as i8);
        out[base + PLAYER_HAS_LARGEST] = f32::from(s.largest_army_player == p as i8);
        out[base + PLAYER_PORT_ANY] = f32::from(s.port_any[p]);
        for r in 0..5 {
            out[base + PLAYER_PORT_2TO1 + r] = f32::from(s.port_resource[p][r]);
        }
        out[base + PLAYER_IS_TURN_OWNER] = f32::from(s.current_player == p);
    }

    // ---- self private ----------------------------------------------------
    for r in 0..5 {
        out[SELF_PRIVATE + r] = s.resources[seat][r] as f32 / 19.0;
        out[SELF_PRIVATE + 5 + r] = s.dev_cards[seat][r] as f32 / 5.0;
        out[SELF_PRIVATE + 10 + r] = s.dev_cards_bought_this_turn[r] as f32 / 2.0;
    }

    // ---- opponents private (Perfect only) --------------------------------
    if visibility == Visibility::Perfect {
        for rel in 1..n {
            let p = (seat + rel) % n;
            let base = OPP_PRIVATE + (rel - 1) * OPP_STRIDE;
            for r in 0..5 {
                out[base + r] = s.resources[p][r] as f32 / 19.0;
                out[base + 5 + r] = s.dev_cards[p][r] as f32 / 5.0;
            }
        }
    }

    // ---- bank -------------------------------------------------------------
    for r in 0..5 {
        out[BANK + r] = s.bank[r] as f32 / 19.0;
    }

    // ---- game context ------------------------------------------------------
    out[CONTEXT + game.game_phase as usize] = 1.0; // 4 slots
    if game.game_phase == GamePhase::Playing {
        out[CONTEXT + 4 + game.turn_phase as usize] = 1.0; // 9 slots
    }
    out[CONTEXT + 13] = s.dice_roll as f32 / 12.0;
    out[CONTEXT + 14] = f32::from(s.has_rolled);
    out[CONTEXT + 15] = s.turn as f32 / 1000.0;
    out[CONTEXT + 16] = target / 10.0;
    out[CONTEXT + 17] = game.roads_to_place as f32 / 2.0;
    out[CONTEXT + 18] =
        (catan_core::game::MAX_TRADES_PER_TURN - game.trades_proposed_this_turn) as f32 / 3.0;
    if game.turn_phase == TurnPhase::RobberDiscard {
        if let Some(&(p, remaining)) = game.pending_discards.get(game.discard_idx) {
            if p == seat {
                out[CONTEXT + 19] = remaining as f32 / 10.0;
            }
        }
    }

    // ---- open trade offer ---------------------------------------------------
    if let Some(offer) = game.trade_offer {
        out[TRADE] = 1.0;
        let rel = rel_of(seat, offer.proposer as usize, n);
        out[TRADE + 1 + rel] = 1.0;
        out[TRADE + 5 + offer.give as usize] = 1.0;
        out[TRADE + 10] = offer.give_amount as f32 / 2.0;
        out[TRADE + 11 + offer.recv as usize] = 1.0;
    }
}
