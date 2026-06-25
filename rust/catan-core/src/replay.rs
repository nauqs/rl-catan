//! Replay support: serialize actions to/from logged records, and the
//! compact binary `GameRecord` format (~2 KB per game) used to log training
//! and evaluation games for later replay/visualization.
//!
//! Numeric action ids are this engine's own log format (they grew out of the
//! retired Python ActionType enum). `dice` and `stolen` carry recorded
//! randomness so replays are fully deterministic.

use crate::game::{Action, CatanGame};

pub fn action_from_log(
    action_type: u8,
    player: u8,
    data: &[i8],
    dice: Option<u8>,
    stolen: Option<i8>,
) -> Action {
    match action_type {
        0 => Action::PlaceInitialSettlement {
            player,
            vertex: data[0] as u8,
        },
        1 => Action::PlaceInitialRoad {
            player,
            edge: data[0] as u8,
        },
        2 => Action::RollDice {
            player,
            forced: dice,
        },
        3 => Action::BuildRoad {
            player,
            edge: data[0] as u8,
        },
        4 => Action::BuildSettlement {
            player,
            vertex: data[0] as u8,
        },
        5 => Action::BuildCity {
            player,
            vertex: data[0] as u8,
        },
        6 => Action::BuyDevCard { player },
        7 => Action::PlayKnight { player },
        8 => Action::PlayRoadBuilding { player },
        9 => Action::PlayYearOfPlenty {
            player,
            r1: data[0] as u8,
            r2: data[1] as u8,
        },
        10 => Action::PlayMonopoly {
            player,
            resource: data[0] as u8,
        },
        11 => Action::MoveRobber {
            player,
            tile: data[0] as u8,
        },
        12 => Action::StealResource {
            player,
            victim: data[0],
            forced: stolen,
        },
        13 => Action::DiscardResource {
            player,
            resource: data[0] as u8,
        },
        14 => Action::TradeWithBank {
            player,
            give: data[0] as u8,
            recv: data[1] as u8,
        },
        15 => Action::ProposeTrade {
            player,
            give: data[0] as u8,
            give_amount: data[1] as u8,
            recv: data[2] as u8,
        },
        16 => Action::RespondTrade {
            player,
            accept: data[0] != 0,
        },
        17 => Action::ConfirmTrade {
            player,
            partner: data[0],
        },
        18 => Action::EndTurn { player },
        other => panic!("unknown action type in log: {other}"),
    }
}

/// Inverse of `action_from_log`: (action_type, player, data).
pub fn action_to_log(action: &Action) -> (u8, u8, Vec<i8>) {
    match *action {
        Action::PlaceInitialSettlement { player, vertex } => (0, player, vec![vertex as i8]),
        Action::PlaceInitialRoad { player, edge } => (1, player, vec![edge as i8]),
        Action::RollDice { player, .. } => (2, player, vec![]),
        Action::BuildRoad { player, edge } => (3, player, vec![edge as i8]),
        Action::BuildSettlement { player, vertex } => (4, player, vec![vertex as i8]),
        Action::BuildCity { player, vertex } => (5, player, vec![vertex as i8]),
        Action::BuyDevCard { player } => (6, player, vec![]),
        Action::PlayKnight { player } => (7, player, vec![]),
        Action::PlayRoadBuilding { player } => (8, player, vec![]),
        Action::PlayYearOfPlenty { player, r1, r2 } => (9, player, vec![r1 as i8, r2 as i8]),
        Action::PlayMonopoly { player, resource } => (10, player, vec![resource as i8]),
        Action::MoveRobber { player, tile } => (11, player, vec![tile as i8]),
        Action::StealResource { player, victim, .. } => (12, player, vec![victim]),
        Action::DiscardResource { player, resource } => (13, player, vec![resource as i8]),
        Action::TradeWithBank { player, give, recv } => (14, player, vec![give as i8, recv as i8]),
        Action::ProposeTrade {
            player,
            give,
            give_amount,
            recv,
        } => (15, player, vec![give as i8, give_amount as i8, recv as i8]),
        Action::RespondTrade { player, accept } => (16, player, vec![accept as i8]),
        Action::ConfirmTrade { player, partner } => (17, player, vec![partner]),
        Action::EndTurn { player } => (18, player, vec![]),
    }
}

// ---------------------------------------------------------------------------
// Binary game records ("CTRP" format)
// ---------------------------------------------------------------------------
//
// Layout (all little-endian):
//   magic "CTRP" (4) | version u8 | num_players u8 | victory_target u8 |
//   reserved u8 | seed u64 | tile_resources [19] | tile_numbers [19] |
//   port_types [9] | dev_deck [25] | winner i8 | turns u16 | final_vp [4] |
//   num_actions u32 | actions...
// Each action: type u8 | player u8 | len u8 | payload [len].
// RollDice payloads carry the rolled total; StealResource payloads carry
// (victim, stolen resource) — randomness is explicit, so replays are exact.

pub const REPLAY_MAGIC: [u8; 4] = *b"CTRP";
pub const REPLAY_VERSION: u8 = 1;

/// A complete recorded game: initial conditions, every action with its
/// randomness made explicit, and an outcome summary (readable without
/// replaying). Typical size: ~2 KB per game.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameRecord {
    pub num_players: u8,
    pub victory_target: u8,
    pub seed: u64,
    pub tile_resources: [i8; 19],
    pub tile_numbers: [i8; 19],
    pub port_types: [i8; 9],
    pub dev_deck: [i8; 25],
    /// (action_type, player, payload) — payload includes recorded randomness.
    pub actions: Vec<(u8, u8, Vec<i8>)>,
    pub winner: i8,
    pub turns: u16,
    pub final_vp: [u8; 4],
}

impl GameRecord {
    /// Begin recording a freshly created game (call before any actions).
    pub fn start(game: &CatanGame) -> GameRecord {
        let s = &game.state;
        GameRecord {
            num_players: s.num_players as u8,
            victory_target: game.state.victory_target as u8,
            seed: s.seed,
            tile_resources: s.tile_resources,
            tile_numbers: s.tile_numbers,
            port_types: s.port_types,
            dev_deck: s.dev_deck,
            actions: Vec::new(),
            winner: -1,
            turns: 0,
            final_vp: [0; 4],
        }
    }

    /// Execute `action` on `game`, appending it (with its randomness made
    /// explicit) to the record. Returns `execute_action`'s result; rejected
    /// actions are not recorded.
    pub fn record_step(&mut self, game: &mut CatanGame, action: &Action) -> bool {
        let steal_before = match *action {
            Action::StealResource { victim, .. } if victim >= 0 => {
                Some((victim as usize, game.state.resources[victim as usize]))
            }
            _ => None,
        };
        if !game.execute_action(action) {
            return false;
        }
        let (t, p, mut data) = action_to_log(action);
        match *action {
            Action::RollDice { .. } => data.push(game.state.dice_roll as i8),
            Action::StealResource { .. } => {
                let stolen = steal_before
                    .and_then(|(victim, before)| {
                        let after = game.state.resources[victim];
                        (0..5).find(|&r| after[r] < before[r]).map(|r| r as i8)
                    })
                    .unwrap_or(-1);
                data.push(stolen);
            }
            _ => {}
        }
        self.actions.push((t, p, data));
        true
    }

    /// Capture the outcome summary once the game is over (or abandoned).
    pub fn finish(&mut self, game: &CatanGame) {
        self.winner = game.winner();
        self.turns = game.state.turn.min(u16::MAX as u32) as u16;
        for p in 0..game.state.num_players {
            self.final_vp[p] = game.state.calculate_victory_points(p) as u8;
        }
    }

    /// Rebuild the game and re-execute every recorded action. Errors if any
    /// action is rejected (i.e. the record does not match the engine).
    pub fn replay(&self) -> Result<CatanGame, String> {
        let mut game = CatanGame::from_replay(
            self.num_players as usize,
            self.tile_resources,
            self.tile_numbers,
            self.port_types,
            self.dev_deck,
        );
        game.state.victory_target = self.victory_target as i32;
        game.record_history = false;
        for (i, (t, p, data)) in self.actions.iter().enumerate() {
            let (core, dice, stolen) = match t {
                2 => (
                    &data[..data.len() - 1],
                    Some(data[data.len() - 1] as u8),
                    None,
                ),
                12 => (&data[..1], None, Some(data[1])),
                _ => (&data[..], None, None),
            };
            let action = action_from_log(*t, *p, core, dice, stolen);
            if !game.execute_action(&action) {
                return Err(format!("action {i} ({action:?}) rejected on replay"));
            }
        }
        Ok(game)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let payload: usize = self.actions.iter().map(|(_, _, d)| 3 + d.len()).sum();
        let mut out = Vec::with_capacity(96 + payload);
        out.extend_from_slice(&REPLAY_MAGIC);
        out.push(REPLAY_VERSION);
        out.push(self.num_players);
        out.push(self.victory_target);
        out.push(0); // reserved
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend(self.tile_resources.iter().map(|&b| b as u8));
        out.extend(self.tile_numbers.iter().map(|&b| b as u8));
        out.extend(self.port_types.iter().map(|&b| b as u8));
        out.extend(self.dev_deck.iter().map(|&b| b as u8));
        out.push(self.winner as u8);
        out.extend_from_slice(&self.turns.to_le_bytes());
        out.extend_from_slice(&self.final_vp);
        out.extend_from_slice(&(self.actions.len() as u32).to_le_bytes());
        for (t, p, data) in &self.actions {
            out.push(*t);
            out.push(*p);
            out.push(data.len() as u8);
            out.extend(data.iter().map(|&b| b as u8));
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<GameRecord, String> {
        struct Cursor<'a> {
            bytes: &'a [u8],
            pos: usize,
        }
        impl<'a> Cursor<'a> {
            fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
                let end = self.pos.checked_add(n).ok_or("overflow")?;
                if end > self.bytes.len() {
                    return Err("replay truncated".into());
                }
                let slice = &self.bytes[self.pos..end];
                self.pos = end;
                Ok(slice)
            }
            fn u8(&mut self) -> Result<u8, String> {
                Ok(self.take(1)?[0])
            }
        }
        fn array<const N: usize>(slice: &[u8]) -> [i8; N] {
            let mut out = [0i8; N];
            for (o, &b) in out.iter_mut().zip(slice) {
                *o = b as i8;
            }
            out
        }

        let mut c = Cursor { bytes, pos: 0 };
        if c.take(4)? != REPLAY_MAGIC {
            return Err("not a CTRP replay (bad magic)".into());
        }
        let version = c.u8()?;
        if version != REPLAY_VERSION {
            return Err(format!("unsupported replay version {version}"));
        }
        let num_players = c.u8()?;
        if !(2..=4).contains(&num_players) {
            return Err(format!("invalid player count {num_players}"));
        }
        let victory_target = c.u8()?;
        c.u8()?; // reserved
        let seed = u64::from_le_bytes(c.take(8)?.try_into().unwrap());
        let tile_resources: [i8; 19] = array(c.take(19)?);
        let tile_numbers: [i8; 19] = array(c.take(19)?);
        let port_types: [i8; 9] = array(c.take(9)?);
        let dev_deck: [i8; 25] = array(c.take(25)?);
        let winner = c.u8()? as i8;
        let turns = u16::from_le_bytes(c.take(2)?.try_into().unwrap());
        let final_vp: [u8; 4] = c.take(4)?.try_into().unwrap();
        let num_actions = u32::from_le_bytes(c.take(4)?.try_into().unwrap()) as usize;

        let mut actions = Vec::with_capacity(num_actions.min(1 << 16));
        for _ in 0..num_actions {
            let t = c.u8()?;
            if t > 18 {
                return Err(format!("invalid action type {t}"));
            }
            let p = c.u8()?;
            let len = c.u8()? as usize;
            let data: Vec<i8> = c.take(len)?.iter().map(|&b| b as i8).collect();
            actions.push((t, p, data));
        }
        if c.pos != bytes.len() {
            return Err("trailing bytes after replay".into());
        }

        Ok(GameRecord {
            num_players,
            victory_target,
            seed,
            tile_resources,
            tile_numbers,
            port_types,
            dev_deck,
            actions,
            winner,
            turns,
            final_vp,
        })
    }
}
