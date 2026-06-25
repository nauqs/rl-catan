//! Fixed discrete action space for RL: every action a policy could ever
//! take has one id in 0..NUM_ACTIONS, valid for the lifetime of a trained
//! network.
//!
//! Design rules:
//! - Ids encode WHAT and WHERE, never WHO: the actor is always the player
//!   the engine says must act (`game.current_player()`).
//! - Seat references (steal victims, trade partners) are RELATIVE to the
//!   actor (+1, +2, +3 around the table) so one policy generalizes across
//!   seats.
//! - Phase-exclusive actions share ids: "place a settlement at vertex v" is
//!   one id whether it's the setup draft or the main phase; "place a road
//!   at edge e" covers setup roads, paid roads, and Road Building card
//!   placements. They can never coexist, and collapsing them keeps the
//!   space tight.
//!
//! Layout (start offset, count):
//!   settlement at vertex      0   54
//!   city at vertex           54   54
//!   road at edge            108   72
//!   robber to tile          180   19
//!   steal rel seat 1-3      199    3
//!   steal skip (nobody)     202    1
//!   discard resource        203    5
//!   monopoly resource       208    5
//!   year of plenty pair     213   15   (r1 <= r2)
//!   bank trade give/recv    228   20   (give != recv)
//!   propose give/amt/recv   248   40   (amount 1-2, give != recv)
//!   respond accept/reject   288    2
//!   confirm rel partner 1-3 290    3
//!   confirm cancel          293    1
//!   roll dice               294    1
//!   buy dev card            295    1
//!   play knight             296    1
//!   play road building      297    1
//!   end turn                298    1

use catan_core::game::{Action, CatanGame, GamePhase};

pub const NUM_ACTIONS: usize = 299;

/// Version of the action-space layout. Trained checkpoints store this and
/// must refuse to load against a different version: changing any id
/// reassigns the meaning of the network's output neurons. Bump on ANY
/// layout change, alongside the layout spot-check tests.
pub const CODEC_VERSION: u32 = 1;

const SETTLEMENT: usize = 0;
const CITY: usize = 54;
const ROAD: usize = 108;
const ROBBER: usize = 180;
const STEAL: usize = 199;
const STEAL_SKIP: usize = 202;
const DISCARD: usize = 203;
const MONOPOLY: usize = 208;
const YOP: usize = 213;
const BANK_TRADE: usize = 228;
const PROPOSE: usize = 248;
const RESPOND: usize = 288;
const CONFIRM: usize = 290;
const CONFIRM_CANCEL: usize = 293;
const ROLL: usize = 294;
const BUY_DEV: usize = 295;
const KNIGHT: usize = 296;
const ROAD_BUILDING: usize = 297;
const END_TURN: usize = 298;

/// Seat offset of `other` from `actor`, going clockwise: 1..=n-1.
fn rel_seat(actor: usize, other: usize, n: usize) -> usize {
    (other + n - actor) % n
}

/// Seat `rel` places clockwise from `actor`. For rel >= n this wraps onto
/// the actor itself, producing an action the engine always rejects — those
/// ids are simply never legal in smaller games.
fn abs_seat(actor: usize, rel: usize, n: usize) -> usize {
    (actor + rel) % n
}

/// Index of an unordered resource pair (r1 <= r2) among the 15 pairs.
fn yop_index(r1: usize, r2: usize) -> usize {
    debug_assert!(r1 <= r2 && r2 < 5);
    r1 * (11 - r1) / 2 + (r2 - r1)
}

fn yop_from_index(idx: usize) -> (u8, u8) {
    let mut base = 0;
    for r1 in 0..5 {
        let row = 5 - r1;
        if idx < base + row {
            return (r1 as u8, (r1 + idx - base) as u8);
        }
        base += row;
    }
    unreachable!("yop index out of range")
}

/// Index of an ordered (give, recv) pair with give != recv: give*4 + recv',
/// where recv' skips the give slot.
fn pair_index(give: usize, recv: usize) -> usize {
    debug_assert!(give != recv && give < 5 && recv < 5);
    give * 4 + if recv < give { recv } else { recv - 1 }
}

fn pair_from_index(idx: usize) -> (u8, u8) {
    let give = idx / 4;
    let r = idx % 4;
    let recv = if r < give { r } else { r + 1 };
    (give as u8, recv as u8)
}

/// Map an engine action to its fixed id. The action must belong to the
/// current actor (which is what `fill_valid_actions` produces).
pub fn encode_action(game: &CatanGame, action: &Action) -> usize {
    let n = game.state.num_players;
    let actor = game.current_player();
    match *action {
        Action::PlaceInitialSettlement { vertex, .. } | Action::BuildSettlement { vertex, .. } => {
            SETTLEMENT + vertex as usize
        }
        Action::BuildCity { vertex, .. } => CITY + vertex as usize,
        Action::PlaceInitialRoad { edge, .. } | Action::BuildRoad { edge, .. } => {
            ROAD + edge as usize
        }
        Action::MoveRobber { tile, .. } => ROBBER + tile as usize,
        Action::StealResource { victim, .. } => {
            if victim < 0 {
                STEAL_SKIP
            } else {
                let rel = rel_seat(actor, victim as usize, n);
                debug_assert!((1..=3).contains(&rel), "steal victim must be another seat");
                STEAL + rel - 1
            }
        }
        Action::DiscardResource { resource, .. } => DISCARD + resource as usize,
        Action::PlayMonopoly { resource, .. } => MONOPOLY + resource as usize,
        Action::PlayYearOfPlenty { r1, r2, .. } => YOP + yop_index(r1 as usize, r2 as usize),
        Action::TradeWithBank { give, recv, .. } => {
            BANK_TRADE + pair_index(give as usize, recv as usize)
        }
        Action::ProposeTrade {
            give,
            give_amount,
            recv,
            ..
        } => {
            PROPOSE
                + give as usize * 8
                + (give_amount as usize - 1) * 4
                + pair_index(give as usize, recv as usize) % 4
        }
        Action::RespondTrade { accept, .. } => RESPOND + usize::from(!accept),
        Action::ConfirmTrade { partner, .. } => {
            if partner < 0 {
                CONFIRM_CANCEL
            } else {
                let rel = rel_seat(actor, partner as usize, n);
                debug_assert!((1..=3).contains(&rel), "trade partner must be another seat");
                CONFIRM + rel - 1
            }
        }
        Action::RollDice { .. } => ROLL,
        Action::BuyDevCard { .. } => BUY_DEV,
        Action::PlayKnight { .. } => KNIGHT,
        Action::PlayRoadBuilding { .. } => ROAD_BUILDING,
        Action::EndTurn { .. } => END_TURN,
    }
}

/// Map an id back to a concrete engine action for the current actor and
/// phase. Total: every id in 0..NUM_ACTIONS decodes to SOME action in every
/// state; ids that make no sense right now decode to actions the engine
/// rejects (and the mask marks illegal).
pub fn decode_action(game: &CatanGame, id: usize) -> Action {
    assert!(id < NUM_ACTIONS, "action id {id} out of range");
    let n = game.state.num_players;
    let actor = game.current_player();
    let player = actor as u8;
    let in_setup = matches!(
        game.game_phase,
        GamePhase::SetupForward | GamePhase::SetupBackward
    );

    if id < CITY {
        let vertex = (id - SETTLEMENT) as u8;
        if in_setup {
            Action::PlaceInitialSettlement { player, vertex }
        } else {
            Action::BuildSettlement { player, vertex }
        }
    } else if id < ROAD {
        Action::BuildCity {
            player,
            vertex: (id - CITY) as u8,
        }
    } else if id < ROBBER {
        let edge = (id - ROAD) as u8;
        if in_setup {
            Action::PlaceInitialRoad { player, edge }
        } else {
            Action::BuildRoad { player, edge }
        }
    } else if id < STEAL {
        Action::MoveRobber {
            player,
            tile: (id - ROBBER) as u8,
        }
    } else if id < STEAL_SKIP {
        let victim = abs_seat(actor, id - STEAL + 1, n) as i8;
        Action::StealResource {
            player,
            victim,
            forced: None,
        }
    } else if id == STEAL_SKIP {
        Action::StealResource {
            player,
            victim: -1,
            forced: None,
        }
    } else if id < MONOPOLY {
        Action::DiscardResource {
            player,
            resource: (id - DISCARD) as u8,
        }
    } else if id < YOP {
        Action::PlayMonopoly {
            player,
            resource: (id - MONOPOLY) as u8,
        }
    } else if id < BANK_TRADE {
        let (r1, r2) = yop_from_index(id - YOP);
        Action::PlayYearOfPlenty { player, r1, r2 }
    } else if id < PROPOSE {
        let (give, recv) = pair_from_index(id - BANK_TRADE);
        Action::TradeWithBank { player, give, recv }
    } else if id < RESPOND {
        let idx = id - PROPOSE;
        let give = (idx / 8) as u8;
        let give_amount = ((idx / 4) % 2 + 1) as u8;
        let r = idx % 4;
        let recv = if (r as u8) < give {
            r as u8
        } else {
            r as u8 + 1
        };
        Action::ProposeTrade {
            player,
            give,
            give_amount,
            recv,
        }
    } else if id < CONFIRM {
        Action::RespondTrade {
            player,
            accept: id == RESPOND,
        }
    } else if id < CONFIRM_CANCEL {
        let partner = abs_seat(actor, id - CONFIRM + 1, n) as i8;
        Action::ConfirmTrade { player, partner }
    } else if id == CONFIRM_CANCEL {
        Action::ConfirmTrade {
            player,
            partner: -1,
        }
    } else if id == ROLL {
        Action::RollDice {
            player,
            forced: None,
        }
    } else if id == BUY_DEV {
        Action::BuyDevCard { player }
    } else if id == KNIGHT {
        Action::PlayKnight { player }
    } else if id == ROAD_BUILDING {
        Action::PlayRoadBuilding { player }
    } else {
        Action::EndTurn { player }
    }
}

/// Fill the legality mask for the current state. `scratch` is a reusable
/// action buffer (zero allocations in steady state).
pub fn fill_action_mask(
    game: &CatanGame,
    scratch: &mut Vec<Action>,
    mask: &mut [bool; NUM_ACTIONS],
) {
    mask.fill(false);
    game.fill_valid_actions(scratch);
    for action in scratch.iter() {
        let id = encode_action(game, action);
        debug_assert!(!mask[id], "two legal actions collided on id {id}");
        mask[id] = true;
    }
}
