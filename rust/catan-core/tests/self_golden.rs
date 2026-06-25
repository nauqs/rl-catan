//! Self-golden regression tests: recorded games (with their randomness made
//! explicit) must replay to the exact same final state. This is the net that
//! catches behavior drift during optimization work.
//!
//! Regenerate after INTENTIONAL rule changes with:
//!   cargo run -p catan-sim --release -- --games 60 --record-golden catan-core/tests/golden/self

use catan_core::game::CatanGame;
use catan_core::replay::action_from_log;
use serde_json::Value;
use std::path::PathBuf;

fn golden_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/self"))
}

fn vec_i8(v: &Value) -> Vec<i8> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i8)
        .collect()
}

fn vec_i16(v: &Value) -> Vec<i16> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i16)
        .collect()
}

fn replay_and_check(path: &std::path::Path) {
    let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let name = path.file_name().unwrap().to_string_lossy().to_string();

    let num_players = data["num_players"].as_u64().unwrap() as usize;
    let mut game = CatanGame::from_replay(
        num_players,
        vec_i8(&data["tile_resources"]).try_into().unwrap(),
        vec_i8(&data["tile_numbers"]).try_into().unwrap(),
        vec_i8(&data["port_types"]).try_into().unwrap(),
        vec_i8(&data["dev_deck"]).try_into().unwrap(),
    );
    game.record_history = false;

    for (i, a) in data["actions"].as_array().unwrap().iter().enumerate() {
        let action = action_from_log(
            a["t"].as_u64().unwrap() as u8,
            a["p"].as_u64().unwrap() as u8,
            &vec_i8(&a["d"]),
            a.get("dice").and_then(|d| d.as_u64()).map(|d| d as u8),
            a.get("stolen").and_then(|s| s.as_i64()).map(|s| s as i8),
        );
        assert!(
            game.execute_action(&action),
            "{name}: action {i} ({action:?}) rejected on replay"
        );
    }

    let f = &data["final"];
    let s = &game.state;
    assert_eq!(
        s.vertices.to_vec(),
        vec_i8(&f["vertices"]),
        "{name}: vertices"
    );
    assert_eq!(s.edges.to_vec(), vec_i8(&f["edges"]), "{name}: edges");
    assert_eq!(s.bank.to_vec(), vec_i16(&f["bank"]), "{name}: bank");
    for p in 0..num_players {
        assert_eq!(
            s.resources[p].to_vec(),
            vec_i16(&f["resources"][p]),
            "{name}: res p{p}"
        );
        assert_eq!(
            s.dev_cards[p].to_vec(),
            vec_i8(&f["dev_cards"][p]),
            "{name}: dev p{p}"
        );
        assert_eq!(
            s.calculate_victory_points(p) as i64,
            f["victory_points"][p].as_i64().unwrap(),
            "{name}: VP p{p}"
        );
        assert_eq!(
            s.knights_played[p] as i64,
            f["knights_played"][p].as_i64().unwrap(),
            "{name}: knights p{p}"
        );
        assert_eq!(
            s.settlements_built[p] as i64,
            f["settlements_built"][p].as_i64().unwrap(),
            "{name}: settlements p{p}"
        );
        assert_eq!(
            s.cities_built[p] as i64,
            f["cities_built"][p].as_i64().unwrap(),
            "{name}: cities p{p}"
        );
        assert_eq!(
            s.roads_built[p] as i64,
            f["roads_built"][p].as_i64().unwrap(),
            "{name}: roads p{p}"
        );
    }
    for (field, value) in [
        ("robber_tile", s.robber_tile as i64),
        ("longest_road_player", s.longest_road_player as i64),
        ("longest_road_length", s.longest_road_length as i64),
        ("largest_army_player", s.largest_army_player as i64),
        ("largest_army_size", s.largest_army_size as i64),
        ("dev_deck_idx", s.dev_deck_idx as i64),
        ("winner", s.winner as i64),
        ("turn", s.turn as i64),
        ("current_player", s.current_player as i64),
        ("game_phase", game.game_phase as i64),
        ("turn_phase", game.turn_phase as i64),
    ] {
        assert_eq!(value, f[field].as_i64().unwrap(), "{name}: {field}");
    }
}

#[test]
fn recorded_games_replay_identically() {
    let dir = golden_dir();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("self-goldens missing — run catan-sim --record-golden (see file header)")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    assert!(
        files.len() >= 50,
        "expected at least 50 self-golden games, found {} — regenerate",
        files.len()
    );
    for f in &files {
        replay_and_check(f);
    }
}
