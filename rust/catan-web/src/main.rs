//! Interactive HTTP server for human play-testing the Rust engine.
//!
//! Exposes live `CatanGame` sessions over JSON so the Three.js visualizer
//! can drive the real engine: every state response carries the full god
//! view (all hands, bank, dev deck, phase machinery) plus the engine's own
//! legal-action list with codec ids. Illegal submissions are the point —
//! the engine must reject them and leave the game untouched, and every
//! response includes a 299-id codec/mask/executor consistency sweep so any
//! disagreement between `fill_valid_actions`, the codec, and
//! `execute_action` surfaces immediately in the UI.
#![recursion_limit = "512"]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use catan_core::game::{Action, CatanGame, GamePhase, TurnPhase};
use catan_core::players::{HeuristicPlayer, Player, RandomPlayer};
use catan_env::codec::{self, NUM_ACTIONS};
use rand::Rng;
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;

const RESOURCE_NAMES: [&str; 5] = ["Wheat", "Sheep", "Wood", "Brick", "Stone"];
const DEV_CARD_NAMES: [&str; 5] = [
    "Knight",
    "Victory Point",
    "Road Building",
    "Year of Plenty",
    "Monopoly",
];

/// Bot strategy per seat; `None` is a human seat.
enum Seat {
    Human,
    Random(RandomPlayer),
    Heuristic(HeuristicPlayer),
}

impl Seat {
    fn is_bot(&self) -> bool {
        !matches!(self, Seat::Human)
    }

    fn kind(&self) -> &'static str {
        match self {
            Seat::Human => "human",
            Seat::Random(_) => "random",
            Seat::Heuristic(_) => "heuristic",
        }
    }
}

struct Session {
    game: CatanGame,
    seats: Vec<Seat>,
}

#[derive(Default)]
struct AppState {
    next_id: u64,
    sessions: HashMap<u64, Session>,
    metrics_path: std::path::PathBuf,
}

type Db = Arc<Mutex<AppState>>;

#[tokio::main]
async fn main() {
    // --metrics-file <path>: the JSONL stream the dashboard tails
    // (written by `catan-sim --metrics` today, the PPO trainer later).
    let args: Vec<String> = std::env::args().collect();
    let mut metrics_path = std::path::PathBuf::from("metrics.jsonl");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--metrics-file" {
            i += 1;
            metrics_path = std::path::PathBuf::from(&args[i]);
        }
        i += 1;
    }

    let db: Db = Arc::new(Mutex::new(AppState {
        metrics_path: metrics_path.clone(),
        ..AppState::default()
    }));

    let app = Router::new()
        .route("/dashboard", get(dashboard_page))
        .route("/rustapi/metrics", get(get_metrics))
        .route("/rustapi/games", post(new_game).get(list_games))
        .route("/rustapi/games/:id", get(get_game))
        .route("/rustapi/games/:id/action", post(post_action))
        .route("/rustapi/games/:id/action-id", post(post_action_id))
        .route("/rustapi/games/:id/bot-step", post(post_bot_step))
        .layer(CorsLayer::permissive())
        .with_state(db);

    let addr = "127.0.0.1:5050";
    println!("catan-web listening on http://{addr}");
    println!("dashboard:        http://{addr}/dashboard");
    println!("metrics file:     {}", metrics_path.display());
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn dashboard_page() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("dashboard.html"))
}

/// Incremental read of the metrics JSONL stream: returns parsed events from
/// byte offset `since`, plus the next offset to poll from. Caps each
/// response so huge files stream over several polls (`more: true`).
async fn get_metrics(
    State(db): State<Db>,
    axum::extract::RawQuery(query): axum::extract::RawQuery,
) -> Json<Value> {
    const MAX_CHUNK: u64 = 4 * 1024 * 1024;
    let since: u64 = query
        .as_deref()
        .unwrap_or("")
        .split('&')
        .find_map(|kv| kv.strip_prefix("since="))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let path = db.lock().unwrap().metrics_path.clone();

    let result =
        tokio::task::spawn_blocking(move || -> std::io::Result<(Vec<Value>, u64, bool)> {
            use std::io::{Read, Seek, SeekFrom};
            let mut file = std::fs::File::open(&path)?;
            let len = file.metadata()?.len();
            let start = since.min(len);
            file.seek(SeekFrom::Start(start))?;
            let to_read = (len - start).min(MAX_CHUNK);
            let mut buf = vec![0u8; to_read as usize];
            file.read_exact(&mut buf)?;
            // Only consume up to the last complete line (a writer may be
            // mid-append on the final one).
            let consumed = match buf.iter().rposition(|&b| b == b'\n') {
                Some(pos) => pos + 1,
                None => 0,
            };
            let events: Vec<Value> = buf[..consumed]
                .split(|&b| b == b'\n')
                .filter(|line| !line.is_empty())
                .filter_map(|line| serde_json::from_slice(line).ok())
                .collect();
            let next = start + consumed as u64;
            Ok((events, next, next < len))
        })
        .await
        .unwrap_or_else(|_| Ok((Vec::new(), since, false)));

    match result {
        Ok((events, next, more)) => Json(json!({ "events": events, "next": next, "more": more })),
        // No file yet: report empty so the dashboard waits politely.
        Err(_) => Json(json!({ "events": [], "next": 0, "more": false })),
    }
}

fn err(status: StatusCode, msg: &str) -> (StatusCode, Json<Value>) {
    (status, Json(json!({ "error": msg })))
}

async fn new_game(
    State(db): State<Db>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let num_players = body["num_players"].as_u64().unwrap_or(4) as usize;
    if !(2..=4).contains(&num_players) {
        return Err(err(StatusCode::BAD_REQUEST, "num_players must be 2-4"));
    }
    let seed = body["seed"]
        .as_u64()
        .unwrap_or_else(|| rand::thread_rng().gen());

    // Default: seat 0 human, the rest heuristic bots.
    let mut seats: Vec<Seat> = Vec::with_capacity(num_players);
    for i in 0..num_players {
        let default = if i == 0 { "human" } else { "heuristic" };
        let kind = body["seats"][i].as_str().unwrap_or(default);
        let bot_seed = seed.wrapping_add(i as u64 + 1);
        seats.push(match kind {
            "human" => Seat::Human,
            "random" => Seat::Random(RandomPlayer::new(bot_seed)),
            "heuristic" => Seat::Heuristic(HeuristicPlayer::new(bot_seed)),
            other => {
                return Err(err(
                    StatusCode::BAD_REQUEST,
                    &format!("unknown seat kind '{other}'"),
                ))
            }
        });
    }

    let game = CatanGame::new(num_players, seed);
    let mut state = db.lock().unwrap();
    state.next_id += 1;
    let id = state.next_id;
    state.sessions.insert(id, Session { game, seats });
    let view = state_view(id, &state.sessions[&id]);
    Ok(Json(view))
}

async fn list_games(State(db): State<Db>) -> Json<Value> {
    let state = db.lock().unwrap();
    let games: Vec<Value> = state
        .sessions
        .iter()
        .map(|(id, s)| {
            json!({
                "id": id,
                "num_players": s.game.state.num_players,
                "turn": s.game.state.turn,
                "finished": s.game.is_game_over(),
            })
        })
        .collect();
    Json(json!({ "games": games }))
}

async fn get_game(
    State(db): State<Db>,
    Path(id): Path<u64>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let state = db.lock().unwrap();
    let session = state
        .sessions
        .get(&id)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "no such game"))?;
    Ok(Json(state_view(id, session)))
}

/// Submit an action as JSON. Illegal actions are expected here: the engine
/// must return ok=false and the state must be byte-identical to before.
async fn post_action(
    State(db): State<Db>,
    Path(id): Path<u64>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let action = action_from_json(&body["action"])
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "unparseable action"))?;
    let mut state = db.lock().unwrap();
    let session = state
        .sessions
        .get_mut(&id)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "no such game"))?;
    let ok = session.game.execute_action(&action);
    let mut view = state_view(id, session);
    view["ok"] = json!(ok);
    view["submitted"] = action_to_json(&action);
    Ok(Json(view))
}

/// Submit a raw codec id (0..299). Decodes through the RL action codec and
/// executes — lets you fire every id, legal or not, and watch the verdict.
async fn post_action_id(
    State(db): State<Db>,
    Path(id): Path<u64>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let action_id = body["id"]
        .as_u64()
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "missing id"))? as usize;
    if action_id >= NUM_ACTIONS {
        return Err(err(StatusCode::BAD_REQUEST, "id out of range (0..299)"));
    }
    let mut state = db.lock().unwrap();
    let session = state
        .sessions
        .get_mut(&id)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "no such game"))?;
    let action = codec::decode_action(&session.game, action_id);
    let ok = session.game.execute_action(&action);
    let mut view = state_view(id, session);
    view["ok"] = json!(ok);
    view["submitted"] = action_to_json(&action);
    view["submitted_codec_id"] = json!(action_id);
    Ok(Json(view))
}

/// Let bot seats play until it's a human's turn (or the game ends).
async fn post_bot_step(
    State(db): State<Db>,
    Path(id): Path<u64>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let mut state = db.lock().unwrap();
    let session = state
        .sessions
        .get_mut(&id)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "no such game"))?;

    let mut played = Vec::new();
    let mut valid: Vec<Action> = Vec::with_capacity(128);
    // Cap well above any legal bot streak (full setup + several turns).
    for _ in 0..2000 {
        if session.game.is_game_over() {
            break;
        }
        let actor = session.game.current_player();
        if !session.seats[actor].is_bot() {
            break;
        }
        session.game.fill_valid_actions(&mut valid);
        if valid.is_empty() {
            break;
        }
        let action = match &mut session.seats[actor] {
            Seat::Human => unreachable!(),
            Seat::Random(p) => p.choose_action(&session.game, &valid),
            Seat::Heuristic(p) => p.choose_action(&session.game, &valid),
        };
        let ok = session.game.execute_action(&action);
        if !ok {
            // A bot picked from valid_actions and the engine refused it:
            // that's an engine/mask bug — surface it loudly.
            let mut view = state_view(id, session);
            view["ok"] = json!(false);
            view["bot_actions"] = json!(played);
            view["bot_error"] = json!(format!(
                "engine rejected a bot action listed as valid: {:?}",
                action
            ));
            return Ok(Json(view));
        }
        played.push(json!({ "player": actor, "action": action_to_json(&action) }));
    }

    let mut view = state_view(id, session);
    view["ok"] = json!(true);
    view["bot_actions"] = json!(played);
    Ok(Json(view))
}

fn game_phase_name(p: GamePhase) -> &'static str {
    match p {
        GamePhase::SetupForward => "SetupForward",
        GamePhase::SetupBackward => "SetupBackward",
        GamePhase::Playing => "Playing",
        GamePhase::Finished => "Finished",
    }
}

fn turn_phase_name(p: TurnPhase) -> &'static str {
    match p {
        TurnPhase::PreRoll => "PreRoll",
        TurnPhase::MustRoll => "MustRoll",
        TurnPhase::RobberDiscard => "RobberDiscard",
        TurnPhase::RobberMove => "RobberMove",
        TurnPhase::RobberSteal => "RobberSteal",
        TurnPhase::Main => "Main",
        TurnPhase::RoadBuilding => "RoadBuilding",
        TurnPhase::TradeResponse => "TradeResponse",
        TurnPhase::TradeChoose => "TradeChoose",
    }
}

fn res_name(r: u8) -> &'static str {
    RESOURCE_NAMES.get(r as usize).copied().unwrap_or("?")
}

fn action_to_json(a: &Action) -> Value {
    match *a {
        Action::PlaceInitialSettlement { player, vertex } => {
            json!({"type": "PlaceInitialSettlement", "player": player, "vertex": vertex})
        }
        Action::PlaceInitialRoad { player, edge } => {
            json!({"type": "PlaceInitialRoad", "player": player, "edge": edge})
        }
        Action::RollDice { player, forced } => {
            json!({"type": "RollDice", "player": player, "forced": forced})
        }
        Action::BuildRoad { player, edge } => {
            json!({"type": "BuildRoad", "player": player, "edge": edge})
        }
        Action::BuildSettlement { player, vertex } => {
            json!({"type": "BuildSettlement", "player": player, "vertex": vertex})
        }
        Action::BuildCity { player, vertex } => {
            json!({"type": "BuildCity", "player": player, "vertex": vertex})
        }
        Action::BuyDevCard { player } => json!({"type": "BuyDevCard", "player": player}),
        Action::PlayKnight { player } => json!({"type": "PlayKnight", "player": player}),
        Action::PlayRoadBuilding { player } => {
            json!({"type": "PlayRoadBuilding", "player": player})
        }
        Action::PlayYearOfPlenty { player, r1, r2 } => {
            json!({"type": "PlayYearOfPlenty", "player": player, "r1": r1, "r2": r2})
        }
        Action::PlayMonopoly { player, resource } => {
            json!({"type": "PlayMonopoly", "player": player, "resource": resource})
        }
        Action::MoveRobber { player, tile } => {
            json!({"type": "MoveRobber", "player": player, "tile": tile})
        }
        Action::StealResource {
            player,
            victim,
            forced,
        } => json!({"type": "StealResource", "player": player, "victim": victim, "forced": forced}),
        Action::DiscardResource { player, resource } => {
            json!({"type": "DiscardResource", "player": player, "resource": resource})
        }
        Action::TradeWithBank { player, give, recv } => {
            json!({"type": "TradeWithBank", "player": player, "give": give, "recv": recv})
        }
        Action::ProposeTrade {
            player,
            give,
            give_amount,
            recv,
        } => json!({"type": "ProposeTrade", "player": player, "give": give,
                    "give_amount": give_amount, "recv": recv}),
        Action::RespondTrade { player, accept } => {
            json!({"type": "RespondTrade", "player": player, "accept": accept})
        }
        Action::ConfirmTrade { player, partner } => {
            json!({"type": "ConfirmTrade", "player": player, "partner": partner})
        }
        Action::EndTurn { player } => json!({"type": "EndTurn", "player": player}),
    }
}

fn action_from_json(v: &Value) -> Option<Action> {
    let t = v["type"].as_str()?;
    let player = v["player"].as_u64()? as u8;
    let u8f = |k: &str| v[k].as_u64().map(|x| x as u8);
    Some(match t {
        "PlaceInitialSettlement" => Action::PlaceInitialSettlement {
            player,
            vertex: u8f("vertex")?,
        },
        "PlaceInitialRoad" => Action::PlaceInitialRoad {
            player,
            edge: u8f("edge")?,
        },
        "RollDice" => Action::RollDice {
            player,
            forced: v["forced"].as_u64().map(|x| x as u8),
        },
        "BuildRoad" => Action::BuildRoad {
            player,
            edge: u8f("edge")?,
        },
        "BuildSettlement" => Action::BuildSettlement {
            player,
            vertex: u8f("vertex")?,
        },
        "BuildCity" => Action::BuildCity {
            player,
            vertex: u8f("vertex")?,
        },
        "BuyDevCard" => Action::BuyDevCard { player },
        "PlayKnight" => Action::PlayKnight { player },
        "PlayRoadBuilding" => Action::PlayRoadBuilding { player },
        "PlayYearOfPlenty" => Action::PlayYearOfPlenty {
            player,
            r1: u8f("r1")?,
            r2: u8f("r2")?,
        },
        "PlayMonopoly" => Action::PlayMonopoly {
            player,
            resource: u8f("resource")?,
        },
        "MoveRobber" => Action::MoveRobber {
            player,
            tile: u8f("tile")?,
        },
        "StealResource" => Action::StealResource {
            player,
            victim: v["victim"].as_i64()? as i8,
            forced: v["forced"].as_i64().map(|x| x as i8),
        },
        "DiscardResource" => Action::DiscardResource {
            player,
            resource: u8f("resource")?,
        },
        "TradeWithBank" => Action::TradeWithBank {
            player,
            give: u8f("give")?,
            recv: u8f("recv")?,
        },
        "ProposeTrade" => Action::ProposeTrade {
            player,
            give: u8f("give")?,
            give_amount: u8f("give_amount")?,
            recv: u8f("recv")?,
        },
        "RespondTrade" => Action::RespondTrade {
            player,
            accept: v["accept"].as_bool()?,
        },
        "ConfirmTrade" => Action::ConfirmTrade {
            player,
            partner: v["partner"].as_i64()? as i8,
        },
        "EndTurn" => Action::EndTurn { player },
        _ => return None,
    })
}

fn action_label(a: &Action) -> String {
    match *a {
        Action::PlaceInitialSettlement { vertex, .. } => {
            format!("Place settlement @ vertex {vertex}")
        }
        Action::PlaceInitialRoad { edge, .. } => format!("Place road @ edge {edge}"),
        Action::RollDice { .. } => "Roll dice".into(),
        Action::BuildRoad { edge, .. } => format!("Build road @ edge {edge}"),
        Action::BuildSettlement { vertex, .. } => format!("Build settlement @ vertex {vertex}"),
        Action::BuildCity { vertex, .. } => format!("Build city @ vertex {vertex}"),
        Action::BuyDevCard { .. } => "Buy dev card".into(),
        Action::PlayKnight { .. } => "Play Knight".into(),
        Action::PlayRoadBuilding { .. } => "Play Road Building".into(),
        Action::PlayYearOfPlenty { r1, r2, .. } => {
            format!("Year of Plenty: {} + {}", res_name(r1), res_name(r2))
        }
        Action::PlayMonopoly { resource, .. } => format!("Monopoly: {}", res_name(resource)),
        Action::MoveRobber { tile, .. } => format!("Move robber → tile {tile}"),
        Action::StealResource { victim, .. } => {
            if victim < 0 {
                "Steal: nobody".into()
            } else {
                format!("Steal from player {victim}")
            }
        }
        Action::DiscardResource { resource, .. } => format!("Discard {}", res_name(resource)),
        Action::TradeWithBank { give, recv, .. } => {
            format!("Bank: {} → {}", res_name(give), res_name(recv))
        }
        Action::ProposeTrade {
            give,
            give_amount,
            recv,
            ..
        } => format!(
            "Offer {give_amount} {} for 1 {}",
            res_name(give),
            res_name(recv)
        ),
        Action::RespondTrade { accept, .. } => {
            if accept {
                "Accept trade".into()
            } else {
                "Reject trade".into()
            }
        }
        Action::ConfirmTrade { partner, .. } => {
            if partner < 0 {
                "Cancel trade".into()
            } else {
                format!("Trade with player {partner}")
            }
        }
        Action::EndTurn { .. } => "End turn".into(),
    }
}

/// For every codec id: the mask, the codec decode, and a clone-execute must
/// agree. A ~500-byte clone times 299 ids is microseconds — running the
/// full sweep on every request turns each UI click into a fuzz step.
fn codec_consistency_check(game: &CatanGame) -> Value {
    let mut scratch = Vec::with_capacity(256);
    let mut mask = [false; NUM_ACTIONS];
    codec::fill_action_mask(game, &mut scratch, &mut mask);

    let mut errors: Vec<String> = Vec::new();

    // Every valid action must roundtrip through the codec.
    for action in &scratch {
        let id = codec::encode_action(game, action);
        let back = codec::decode_action(game, id);
        if back != *action {
            errors.push(format!(
                "roundtrip mismatch: {action:?} -> id {id} -> {back:?}"
            ));
        }
    }
    if scratch.len() != mask.iter().filter(|&&m| m).count() {
        errors.push(format!(
            "mask count {} != valid action count {} (id collision)",
            mask.iter().filter(|&&m| m).count(),
            scratch.len()
        ));
    }

    // Mask must equal executor verdict for ALL 299 ids.
    for id in 0..NUM_ACTIONS {
        let action = codec::decode_action(game, id);
        let mut clone = game.clone();
        let executed = clone.execute_action(&action);
        if executed != mask[id] {
            errors.push(format!(
                "id {id} ({action:?}): mask says {} but execute says {}",
                mask[id], executed
            ));
        }
    }

    json!({ "ok": errors.is_empty(), "errors": errors })
}

fn state_view(id: u64, session: &Session) -> Value {
    let game = &session.game;
    let s = &game.state;
    let n = s.num_players;

    let mut valid = Vec::with_capacity(256);
    game.fill_valid_actions(&mut valid);
    let valid_json: Vec<Value> = valid
        .iter()
        .map(|a| {
            json!({
                "codec_id": codec::encode_action(game, a),
                "label": action_label(a),
                "action": action_to_json(a),
            })
        })
        .collect();

    let victory_points: Vec<i32> = (0..n).map(|p| s.calculate_victory_points(p)).collect();

    // Remaining dev deck composition (god view): cards not yet drawn.
    let mut deck_remaining = [0u8; 5];
    for &card in &s.dev_deck[s.dev_deck_idx..] {
        deck_remaining[card as usize] += 1;
    }

    let history_tail: Vec<Value> = game
        .action_history
        .iter()
        .rev()
        .take(30)
        .rev()
        .map(action_to_json)
        .collect();

    json!({
        "id": id,
        "game_phase": game_phase_name(game.game_phase),
        "turn_phase": turn_phase_name(game.turn_phase),
        "current_player": game.current_player(),
        "seats": session.seats.iter().map(|b| b.kind()).collect::<Vec<_>>(),
        "roads_to_place": game.roads_to_place,
        "pending_discards": game.pending_discards,
        "trade_offer": game.trade_offer.map(|t| json!({
            "proposer": t.proposer, "give": t.give,
            "give_amount": t.give_amount, "recv": t.recv,
            "give_name": res_name(t.give), "recv_name": res_name(t.recv),
        })),
        "trade_accepts": game.trade_accepts[..n].to_vec(),
        "trades_proposed_this_turn": game.trades_proposed_this_turn,
        "resource_names": RESOURCE_NAMES,
        "dev_card_names": DEV_CARD_NAMES,
        "state": {
            "seed": s.seed,
            "num_players": n,
            "phase": s.phase,
            "turn": s.turn,
            "current_player": s.current_player,
            "winner": s.winner,
            "victory_target": s.victory_target,
            "tile_resources": s.tile_resources,
            "tile_numbers": s.tile_numbers,
            "port_types": s.port_types,
            "vertices": s.vertices.to_vec(),
            "edges": s.edges.to_vec(),
            "resources": s.resources[..n].to_vec(),
            "bank": s.bank,
            "victory_points": victory_points,
            "robber_tile": s.robber_tile,
            "dice_roll": s.dice_roll,
            "has_rolled": s.has_rolled,
            "longest_road_player": s.longest_road_player,
            "longest_road_length": s.longest_road_length,
            "largest_army_player": s.largest_army_player,
            "largest_army_size": s.largest_army_size,
            "dev_cards": s.dev_cards[..n].to_vec(),
            "knights_played": s.knights_played[..n].to_vec(),
            "dev_deck_remaining": deck_remaining,
            "dev_deck_total_remaining": s.dev_deck.len() - s.dev_deck_idx,
            "settlements_built": s.settlements_built[..n].to_vec(),
            "cities_built": s.cities_built[..n].to_vec(),
            "roads_built": s.roads_built[..n].to_vec(),
            "road_lengths": s.road_lengths[..n].to_vec(),
            "port_any": s.port_any[..n].to_vec(),
            "port_resource": s.port_resource[..n].to_vec(),
        },
        "valid_actions": valid_json,
        "codec_check": codec_consistency_check(game),
        "history_len": game.action_history.len(),
        "history_tail": history_tail,
    })
}
