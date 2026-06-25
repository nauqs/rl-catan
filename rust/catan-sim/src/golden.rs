//! Self-golden recorder: plays games and writes them as JSON fixtures
//! (initial board + dev deck, action stream with recorded randomness, final
//! state). `catan-core/tests/self_golden.rs` replays these and verifies the
//! final state, pinning engine behavior across refactors and optimizations.

use std::path::Path;

use catan_core::game::{Action, CatanGame};
use catan_core::players::{HeuristicPlayer, Player, RandomPlayer};
use catan_core::replay::action_to_log;
use serde_json::json;

fn final_state_json(game: &CatanGame) -> serde_json::Value {
    let s = &game.state;
    let n = s.num_players;
    json!({
        "vertices": s.vertices.to_vec(),
        "edges": s.edges.to_vec(),
        "resources": (0..n).map(|p| s.resources[p].to_vec()).collect::<Vec<_>>(),
        "bank": s.bank.to_vec(),
        "victory_points": (0..n).map(|p| s.calculate_victory_points(p)).collect::<Vec<_>>(),
        "dev_cards": (0..n).map(|p| s.dev_cards[p].to_vec()).collect::<Vec<_>>(),
        "knights_played": s.knights_played[..n].to_vec(),
        "settlements_built": s.settlements_built[..n].to_vec(),
        "cities_built": s.cities_built[..n].to_vec(),
        "roads_built": s.roads_built[..n].to_vec(),
        "robber_tile": s.robber_tile,
        "longest_road_player": s.longest_road_player,
        "longest_road_length": s.longest_road_length,
        "largest_army_player": s.largest_army_player,
        "largest_army_size": s.largest_army_size,
        "dev_deck_idx": s.dev_deck_idx,
        "winner": s.winner,
        "turn": s.turn,
        "current_player": s.current_player,
        "game_phase": game.game_phase as u8,
        "turn_phase": game.turn_phase as u8,
    })
}

fn record_game(seed: u64) -> serde_json::Value {
    let num_players = if seed.is_multiple_of(5) { 3 } else { 4 };
    let mut game = CatanGame::new(num_players, seed);
    game.record_history = false;

    let mut players: Vec<Box<dyn Player>> = (0..num_players as u64)
        .map(|i| -> Box<dyn Player> {
            match (seed % 3, i % 2) {
                (0, _) => Box::new(RandomPlayer::new(seed * 31 + i)),
                (1, _) => Box::new(HeuristicPlayer::new(seed * 31 + i)),
                (_, 0) => Box::new(RandomPlayer::new(seed * 31 + i)),
                (_, _) => Box::new(HeuristicPlayer::new(seed * 31 + i)),
            }
        })
        .collect();

    let initial = json!({
        "seed": seed,
        "num_players": num_players,
        "tile_resources": game.state.tile_resources.to_vec(),
        "tile_numbers": game.state.tile_numbers.to_vec(),
        "port_types": game.state.port_types.to_vec(),
        "dev_deck": game.state.dev_deck.to_vec(),
    });

    let mut actions = Vec::new();
    let mut valid: Vec<Action> = Vec::with_capacity(128);
    while !game.is_game_over() && game.state.turn < 1000 {
        let idx = game.current_player();
        game.fill_valid_actions(&mut valid);
        if valid.is_empty() {
            break;
        }
        let action = players[idx].choose_action(&game, &valid);

        // Snapshot randomness inputs so the replay is deterministic.
        let steal_snapshot = match action {
            Action::StealResource { victim, .. } if victim >= 0 => {
                Some((victim as usize, game.state.resources[victim as usize]))
            }
            _ => None,
        };

        assert!(
            game.execute_action(&action),
            "recorder chose an illegal action"
        );

        let (t, p, d) = action_to_log(&action);
        let mut entry = json!({ "t": t, "p": p, "d": d });
        if matches!(action, Action::RollDice { .. }) {
            entry["dice"] = json!(game.state.dice_roll);
        }
        if let Some((victim, before)) = steal_snapshot {
            let after = game.state.resources[victim];
            let stolen = (0..5)
                .find(|&r| after[r] < before[r])
                .map(|r| r as i8)
                .unwrap_or(-1);
            entry["stolen"] = json!(stolen);
        }
        actions.push(entry);
    }

    let mut record = initial;
    record["actions"] = json!(actions);
    record["final"] = final_state_json(&game);
    record
}

pub fn record_goldens(dir: &Path, games: u64) {
    std::fs::create_dir_all(dir).expect("create golden dir");
    let mut total_actions = 0usize;
    for seed in 1..=games {
        let record = record_game(seed);
        let n = record["actions"].as_array().unwrap().len();
        total_actions += n;
        let path = dir.join(format!("seed_{seed:03}.json"));
        std::fs::write(&path, serde_json::to_string(&record).unwrap()).expect("write golden");
        println!(
            "seed {seed:3}: {}p, {n} actions, winner={}, turns={}",
            record["num_players"], record["final"]["winner"], record["final"]["turn"]
        );
    }
    println!(
        "\nrecorded {games} games, {total_actions} actions into {}",
        dir.display()
    );
}
