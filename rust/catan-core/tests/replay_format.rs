//! The CTRP binary replay format: recorded games must roundtrip through
//! bytes exactly and replay to the same final state; malformed bytes must
//! be rejected, never panic.

use catan_core::game::CatanGame;
use catan_core::players::{HeuristicPlayer, Player, RandomPlayer};
use catan_core::replay::GameRecord;

/// Play a full game while recording it; return the record and the live game.
fn record_full_game(seed: u64, num_players: usize, heuristic: bool) -> (GameRecord, CatanGame) {
    let mut game = CatanGame::new(num_players, seed);
    game.record_history = false;
    let mut players: Vec<Box<dyn Player>> = (0..num_players as u64)
        .map(|i| -> Box<dyn Player> {
            if heuristic && i % 2 == 0 {
                Box::new(HeuristicPlayer::new(seed * 3 + i))
            } else {
                Box::new(RandomPlayer::new(seed * 3 + i))
            }
        })
        .collect();

    let mut record = GameRecord::start(&game);
    let mut valid = Vec::with_capacity(512);
    while !game.is_game_over() && game.state.turn < 1000 {
        game.fill_valid_actions(&mut valid);
        if valid.is_empty() {
            break;
        }
        let idx = game.current_player();
        let action = players[idx].choose_action(&game, &valid);
        assert!(
            record.record_step(&mut game, &action),
            "recorder hit a rejected action"
        );
    }
    record.finish(&game);
    (record, game)
}

#[test]
fn records_roundtrip_through_bytes_and_replay_exactly() {
    let mut total_bytes = 0usize;
    let mut total_actions = 0usize;
    for (seed, players, heuristic) in [(1u64, 4, false), (2, 3, false), (3, 4, true), (4, 3, true)]
    {
        let (record, live) = record_full_game(seed, players, heuristic);

        // Byte roundtrip is exact.
        let bytes = record.to_bytes();
        let decoded = GameRecord::from_bytes(&bytes).expect("decode");
        assert_eq!(
            decoded, record,
            "seed {seed}: byte roundtrip changed the record"
        );
        total_bytes += bytes.len();
        total_actions += record.actions.len();

        // Replaying reaches the identical final state.
        let replayed = decoded.replay().expect("replay");
        assert_eq!(
            replayed.state.vertices, live.state.vertices,
            "seed {seed}: vertices"
        );
        assert_eq!(replayed.state.edges, live.state.edges, "seed {seed}: edges");
        assert_eq!(
            replayed.state.resources, live.state.resources,
            "seed {seed}: hands"
        );
        assert_eq!(replayed.state.bank, live.state.bank, "seed {seed}: bank");
        assert_eq!(replayed.winner(), live.winner(), "seed {seed}: winner");
        assert_eq!(replayed.state.turn, live.state.turn, "seed {seed}: turns");

        // The summary header matches reality (readable without replaying).
        assert_eq!(record.winner, live.winner());
        for p in 0..players {
            assert_eq!(
                record.final_vp[p] as i32,
                live.state.calculate_victory_points(p),
                "seed {seed}: final VP p{p}"
            );
        }
    }
    let per_action = total_bytes as f64 / total_actions as f64;
    println!("{total_actions} actions in {total_bytes} bytes ({per_action:.2} bytes/action)");
    assert!(
        per_action < 8.0,
        "format bloated: {per_action:.2} bytes/action"
    );
}

#[test]
fn malformed_bytes_are_rejected_not_panics() {
    let (record, _) = record_full_game(7, 4, false);
    let good = record.to_bytes();

    // Bad magic.
    let mut bad = good.clone();
    bad[0] = b'X';
    assert!(GameRecord::from_bytes(&bad).is_err());

    // Bad version.
    let mut bad = good.clone();
    bad[4] = 99;
    assert!(GameRecord::from_bytes(&bad).is_err());

    // Every truncation point must error cleanly (never panic).
    for cut in 0..good.len().min(120) {
        assert!(
            GameRecord::from_bytes(&good[..cut]).is_err(),
            "truncation at {cut} accepted"
        );
    }
    assert!(
        GameRecord::from_bytes(&good[..good.len() - 1]).is_err(),
        "tail truncation"
    );

    // Trailing garbage rejected.
    let mut bad = good.clone();
    bad.push(0);
    assert!(GameRecord::from_bytes(&bad).is_err());

    // The original still decodes.
    assert!(GameRecord::from_bytes(&good).is_ok());
}

#[test]
fn unfinished_games_record_cleanly() {
    // Records of abandoned games (e.g. mid-training truncation) still
    // roundtrip and replay.
    let mut game = CatanGame::new(4, 11);
    game.record_history = false;
    let mut players: Vec<RandomPlayer> = (0..4u64).map(|i| RandomPlayer::new(50 + i)).collect();
    let mut record = GameRecord::start(&game);
    let mut valid = Vec::with_capacity(256);
    for _ in 0..200 {
        game.fill_valid_actions(&mut valid);
        let idx = game.current_player();
        let action = players[idx].choose_action(&game, &valid);
        assert!(record.record_step(&mut game, &action));
    }
    record.finish(&game);
    assert_eq!(record.winner, -1, "no winner yet");

    let decoded = GameRecord::from_bytes(&record.to_bytes()).unwrap();
    let replayed = decoded.replay().unwrap();
    assert_eq!(replayed.state.vertices, game.state.vertices);
    assert_eq!(replayed.turn_phase, game.turn_phase);
}

#[test]
fn custom_victory_target_survives_the_replay_roundtrip() {
    use catan_core::players::{Player, RandomPlayer};
    let mut game = CatanGame::new_with_target(4, 19, 7);
    game.record_history = false;
    let mut players: Vec<RandomPlayer> = (0..4u64).map(|i| RandomPlayer::new(60 + i)).collect();
    let mut record = GameRecord::start(&game);
    assert_eq!(record.victory_target, 7);
    let mut valid = Vec::with_capacity(256);
    while !game.is_game_over() && game.state.turn < 1000 {
        game.fill_valid_actions(&mut valid);
        let idx = game.current_player();
        let action = players[idx].choose_action(&game, &valid);
        assert!(record.record_step(&mut game, &action));
    }
    record.finish(&game);
    let decoded = GameRecord::from_bytes(&record.to_bytes()).unwrap();
    assert_eq!(decoded.victory_target, 7);
    // The replay must enforce the SAME threshold: identical final state.
    let replayed = decoded.replay().unwrap();
    assert_eq!(replayed.state.victory_target, 7);
    assert_eq!(replayed.winner(), game.winner());
    assert_eq!(replayed.state.turn, game.state.turn);
    if game.winner() >= 0 {
        let vp = replayed
            .state
            .calculate_victory_points(game.winner() as usize);
        assert!(
            (7..10).contains(&vp),
            "first-to-7 winner should win at 7-9 VP, got {vp}"
        );
    }
}
