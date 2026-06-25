//! Mass Catan simulation CLI — the Rust counterpart of run_simulation.py.
//!
//! Usage:
//!   catan-sim --games 100000 --players R,R,R,R [--seed 42] [--single-thread]

use std::time::Instant;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use catan_core::eval::GreedyValuePlayer;
use catan_core::game::{CatanGame, GamePhase};
use catan_core::players::{HeuristicPlayer, Player, RandomPlayer, RolloutBot};

mod evolve;
mod golden;

#[derive(Clone, Copy, PartialEq)]
enum Strategy {
    Random,
    Heuristic,
    HeuristicV2,
    Rollout,
    GreedyValue,
    Alpha,
}

impl Strategy {
    fn name(self) -> &'static str {
        match self {
            Strategy::Random => "RandomPlayer",
            Strategy::Heuristic => "HeuristicPlayer",
            Strategy::HeuristicV2 => "HeuristicV2",
            Strategy::Rollout => "RolloutBot",
            Strategy::Alpha => "AlphaBot",
            Strategy::GreedyValue => "GreedyValue",
        }
    }
}

struct Config {
    games: usize,
    strategies: Vec<Strategy>,
    seed: u64,
    single_thread: bool,
    profile_steps: bool,
    record_replays: Option<std::path::PathBuf>,
    metrics: Option<std::path::PathBuf>,
    rollout_cfg: (usize, u32),
    net: Option<std::sync::Arc<catan_env::net::MlpNet>>,
    alpha_cfg: (usize, usize, u32),
}

#[derive(Clone, Copy)]
struct GameOutcome {
    winner: i8,
    turns: u32,
    steps: u64,
}

fn parse_args() -> Config {
    let mut games = 1000usize;
    let mut strategies = vec![Strategy::Random; 4];
    let mut seed = 0u64;
    let mut single_thread = false;
    let mut profile_steps = false;
    let mut record_replays: Option<std::path::PathBuf> = None;
    let mut rollout_cfg = (12usize, 40u32);
    let mut net_path: Option<std::path::PathBuf> = None;
    let mut alpha_cfg = (10usize, 4usize, 12u32); // root_k, samples, depth
    let mut metrics: Option<std::path::PathBuf> = None;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" | "-n" => {
                i += 1;
                games = args[i].parse().expect("invalid --games");
            }
            "--players" | "-p" => {
                i += 1;
                strategies = args[i]
                    .split(',')
                    .map(|s| match s.trim().to_uppercase().as_str() {
                        "R" | "RANDOM" => Strategy::Random,
                        "H" | "HEURISTIC" => Strategy::Heuristic,
                        "V" | "HEURISTIC_V2" => Strategy::HeuristicV2,
                        "O" | "ROLLOUT" => Strategy::Rollout,
                        "A" | "ALPHA" => Strategy::Alpha,
                        "G" | "GREEDY" => Strategy::GreedyValue,
                        other => panic!("unknown player type: {other}"),
                    })
                    .collect();
                assert!((2..=4).contains(&strategies.len()), "need 2-4 players");
            }
            "--seed" | "-s" => {
                i += 1;
                seed = args[i].parse().expect("invalid --seed");
            }
            "--single-thread" => single_thread = true,
            "--profile-steps" => profile_steps = true,
            "--net" => {
                i += 1;
                net_path = Some(std::path::PathBuf::from(&args[i]));
            }
            "--alpha-config" => {
                i += 1;
                let parts: Vec<&str> = args[i].split(',').collect();
                alpha_cfg = (
                    parts[0].parse().unwrap(),
                    parts[1].parse().unwrap(),
                    parts[2].parse().unwrap(),
                );
            }
            "--rollout-config" => {
                i += 1;
                let parts: Vec<&str> = args[i].split(',').collect();
                rollout_cfg = (parts[0].parse().unwrap(), parts[1].parse().unwrap());
            }
            "--record-replays" => {
                i += 1;
                record_replays = Some(std::path::PathBuf::from(&args[i]));
            }
            "--metrics" => {
                i += 1;
                metrics = Some(std::path::PathBuf::from(&args[i]));
            }
            "--dump-replay" => {
                i += 1;
                dump_replay(std::path::Path::new(&args[i]));
                std::process::exit(0);
            }
            "--evolve" => {
                evolve::evolve(games as u64, seed);
                std::process::exit(0);
            }
            "--record-golden" => {
                i += 1;
                let dir = std::path::PathBuf::from(&args[i]);
                golden::record_goldens(&dir, games as u64);
                std::process::exit(0);
            }
            "--help" | "-h" => {
                println!(
                    "catan-sim --games N --players R,R,H,H [--seed S] [--single-thread]\n\
                     catan-sim ... --record-replays <dir>        (write a .ctrp replay per game)\n\
                     catan-sim --dump-replay <file.ctrp>         (print a replay as JSON)\n\
                     catan-sim --games N --record-golden <dir>   (regenerate test fixtures)"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }
    Config {
        games,
        strategies,
        seed,
        single_thread,
        profile_steps,
        record_replays,
        metrics,
        rollout_cfg,
        net: net_path.map(|p| std::sync::Arc::new(catan_env::net::MlpNet::load(&p))),
        alpha_cfg,
    }
}

/// Live metrics stream: one JSON line per event, appended as games finish.
/// The catan-web dashboard tails this file; the PPO trainer will append
/// `train` and `eval` events to the same stream (schema in training/README).
struct MetricsWriter {
    inner: std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
}

impl MetricsWriter {
    fn create(path: &std::path::Path) -> MetricsWriter {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = std::fs::File::create(path).expect("create metrics file");
        MetricsWriter {
            inner: std::sync::Mutex::new(std::io::BufWriter::new(file)),
        }
    }

    fn emit(&self, value: serde_json::Value) {
        use std::io::Write;
        let mut w = self.inner.lock().unwrap();
        serde_json::to_writer(&mut *w, &value).unwrap();
        w.write_all(b"\n").unwrap();
        w.flush().unwrap();
    }
}

/// Print a CTRP replay as JSON (header summary + decoded action stream).
fn dump_replay(path: &std::path::Path) {
    const ACTION_NAMES: [&str; 19] = [
        "place_initial_settlement",
        "place_initial_road",
        "roll_dice",
        "build_road",
        "build_settlement",
        "build_city",
        "buy_dev_card",
        "play_knight",
        "play_road_building",
        "play_year_of_plenty",
        "play_monopoly",
        "move_robber",
        "steal_resource",
        "discard_resource",
        "trade_with_bank",
        "propose_trade",
        "respond_trade",
        "confirm_trade",
        "end_turn",
    ];
    let bytes = std::fs::read(path).expect("read replay file");
    let record = catan_core::replay::GameRecord::from_bytes(&bytes).expect("parse replay");
    // Verify it still replays before printing.
    record.replay().expect("replay verification failed");
    let actions: Vec<serde_json::Value> = record
        .actions
        .iter()
        .map(|(t, p, d)| {
            serde_json::json!({
                "name": ACTION_NAMES[*t as usize],
                "player": p,
                "data": d,
            })
        })
        .collect();
    let out = serde_json::json!({
        "num_players": record.num_players,
        "victory_target": record.victory_target,
        "seed": record.seed,
        "tile_resources": record.tile_resources.to_vec(),
        "tile_numbers": record.tile_numbers.to_vec(),
        "port_types": record.port_types.to_vec(),
        "winner": record.winner,
        "turns": record.turns,
        "final_vp": record.final_vp.to_vec(),
        "num_actions": record.actions.len(),
        "actions": actions,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

fn make_players(
    game_seed: u64,
    strategies: &[Strategy],
    rollout_cfg: (usize, u32),
    net: &Option<std::sync::Arc<catan_env::net::MlpNet>>,
    alpha_cfg: (usize, usize, u32),
) -> Vec<Box<dyn Player>> {
    let mut seed_rng = SmallRng::seed_from_u64(game_seed);
    strategies
        .iter()
        .map(|s| -> Box<dyn Player> {
            let player_seed = seed_rng.gen::<u64>();
            match s {
                Strategy::Random => Box::new(RandomPlayer::new(player_seed)),
                Strategy::Heuristic => Box::new(HeuristicPlayer::new(player_seed)),
                Strategy::HeuristicV2 => Box::new(HeuristicPlayer::v2(player_seed)),
                Strategy::Rollout => Box::new(RolloutBot::new(player_seed, rollout_cfg.0, rollout_cfg.1)),
                Strategy::Alpha => Box::new(catan_env::alpha::AlphaBot::new(
                    player_seed,
                    net.as_ref().expect("--net <file.ctnn> required for A seats").clone(),
                    alpha_cfg.0,
                    alpha_cfg.1,
                    alpha_cfg.2,
                )),
                Strategy::GreedyValue => Box::new(GreedyValuePlayer::new(player_seed)),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    game_seed: u64,
    game_idx: usize,
    strategies: &[Strategy],
    record_dir: Option<&std::path::Path>,
    metrics: Option<&MetricsWriter>,
    rollout_cfg: (usize, u32),
    net: &Option<std::sync::Arc<catan_env::net::MlpNet>>,
    alpha_cfg: (usize, usize, u32),
) -> GameOutcome {
    let mut game = CatanGame::new(strategies.len(), game_seed);
    game.record_history = false; // bulk simulation never reads the history
    let mut players = make_players(game_seed, strategies, rollout_cfg, net, alpha_cfg);
    let mut record = record_dir.map(|_| catan_core::replay::GameRecord::start(&game));
    let mut steps = 0u64;
    let mut valid = Vec::with_capacity(128);
    while !game.is_game_over() && game.state.turn < 1000 {
        let idx = game.current_player();
        game.fill_valid_actions(&mut valid);
        if valid.is_empty() {
            break;
        }
        let action = players[idx].choose_action(&game, &valid);
        match record.as_mut() {
            Some(rec) => {
                rec.record_step(&mut game, &action);
            }
            None => {
                game.execute_action(&action);
            }
        }
        steps += 1;
    }
    if let (Some(mut rec), Some(dir)) = (record, record_dir) {
        rec.finish(&game);
        let path = dir.join(format!("game_{game_idx:06}_{game_seed}.ctrp"));
        std::fs::write(&path, rec.to_bytes()).expect("write replay");
    }
    if let Some(m) = metrics {
        let n = game.state.num_players;
        let vp: Vec<i32> = (0..n)
            .map(|p| game.state.calculate_victory_points(p))
            .collect();
        m.emit(serde_json::json!({
            "t": "game",
            "i": game_idx,
            "winner": game.winner(),
            "turns": game.state.turn,
            "steps": steps,
            "vp": vp,
            "cap": game.state.turn >= 1000,
            "unix_ms": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }));
    }
    GameOutcome {
        winner: game.winner(),
        turns: game.state.turn,
        steps,
    }
}

const PHASE_NAMES: [&str; 10] = [
    "pre_roll",
    "must_roll",
    "robber_discard",
    "robber_move",
    "robber_steal",
    "main",
    "road_building",
    "trade_response",
    "trade_choose",
    "setup",
];

/// Single-threaded per-step timing: how long the ENGINE takes to fulfill a
/// state transition (legal-mask generation + action execution) vs how long
/// the AGENT takes to decide. The engine share is the optimization target;
/// the agent share gets replaced by a neural network in RL.
fn profile_step_breakdown(config: &Config) {
    let n = config.strategies.len();

    // Calibrate timer overhead (4 Instant::now() calls per step).
    let cal_n = 1_000_000u32;
    let cal_start = Instant::now();
    for _ in 0..cal_n {
        std::hint::black_box(Instant::now());
    }
    let timer_ns = cal_start.elapsed().as_nanos() as f64 / cal_n as f64;

    let mut seed_rng = SmallRng::seed_from_u64(config.seed);
    let (mut mask_ns, mut agent_ns, mut exec_ns) = (0u128, 0u128, 0u128);
    let mut steps = 0u64;
    let mut phase_ns = [0u128; 10];
    let mut phase_steps = [0u64; 10];

    for _ in 0..config.games {
        let game_seed = seed_rng.gen();
        let mut game = CatanGame::new(n, game_seed);
        game.record_history = false;
        let mut players = make_players(game_seed, &config.strategies, config.rollout_cfg, &config.net, config.alpha_cfg);
        let mut valid = Vec::with_capacity(128);
        while !game.is_game_over() && game.state.turn < 1000 {
            let phase_idx = match game.game_phase {
                GamePhase::Playing => game.turn_phase as usize,
                _ => 9,
            };
            let idx = game.current_player();
            let t0 = Instant::now();
            game.fill_valid_actions(&mut valid);
            let t1 = Instant::now();
            if valid.is_empty() {
                break;
            }
            let action = players[idx].choose_action(&game, &valid);
            let t2 = Instant::now();
            game.execute_action(&action);
            let t3 = Instant::now();

            let mask = (t1 - t0).as_nanos();
            let exec = (t3 - t2).as_nanos();
            mask_ns += mask;
            agent_ns += (t2 - t1).as_nanos();
            exec_ns += exec;
            phase_ns[phase_idx] += mask + exec;
            phase_steps[phase_idx] += 1;
            steps += 1;
        }
    }

    let per = |ns: u128| ns as f64 / steps as f64;
    let engine = per(mask_ns) + per(exec_ns);
    let total = engine + per(agent_ns);
    println!(
        "=== per-step breakdown ({} games, {} steps, 1 thread) ===",
        config.games, steps
    );
    println!(
        "engine: legal mask    {:8.0} ns/step ({:4.1}%)",
        per(mask_ns),
        per(mask_ns) / total * 100.0
    );
    println!(
        "engine: execute       {:8.0} ns/step ({:4.1}%)",
        per(exec_ns),
        per(exec_ns) / total * 100.0
    );
    println!(
        "agent:  choose        {:8.0} ns/step ({:4.1}%)",
        per(agent_ns),
        per(agent_ns) / total * 100.0
    );
    println!(
        "total                 {:8.0} ns/step -> {:.0} steps/sec",
        total,
        1e9 / total
    );
    println!(
        "engine-only ceiling   {:8.0} ns/step -> {:.0} steps/sec (if agent were free)",
        engine,
        1e9 / engine
    );
    println!(
        "timer overhead ~{:.0} ns/step included in the buckets above",
        4.0 * timer_ns
    );
    println!(
        "
engine time by phase (mask + execute):"
    );
    println!(
        "{:<16} {:>10} {:>8} {:>12} {:>8}",
        "phase", "steps", "steps%", "ns/step", "time%"
    );
    let total_phase: u128 = phase_ns.iter().sum();
    for i in 0..10 {
        if phase_steps[i] == 0 {
            continue;
        }
        println!(
            "{:<16} {:>10} {:>7.1}% {:>12.0} {:>7.1}%",
            PHASE_NAMES[i],
            phase_steps[i],
            phase_steps[i] as f64 / steps as f64 * 100.0,
            phase_ns[i] as f64 / phase_steps[i] as f64,
            phase_ns[i] as f64 / total_phase as f64 * 100.0
        );
    }
}

fn main() {
    let config = parse_args();
    let n = config.strategies.len();

    if config.profile_steps {
        profile_step_breakdown(&config);
        return;
    }

    // Pre-derive per-game seeds so results are reproducible regardless of
    // scheduling order.
    let mut seed_rng = SmallRng::seed_from_u64(config.seed);
    let seeds: Vec<u64> = (0..config.games).map(|_| seed_rng.gen()).collect();

    let start = Instant::now();
    if let Some(dir) = &config.record_replays {
        std::fs::create_dir_all(dir).expect("create replay dir");
    }
    let record_dir = config.record_replays.as_deref();
    let metrics = config.metrics.as_deref().map(MetricsWriter::create);
    if let Some(m) = &metrics {
        m.emit(serde_json::json!({
            "t": "run",
            "source": "catan-sim",
            "games": config.games,
            "players": config.strategies.iter().map(|s| s.name()).collect::<Vec<_>>(),
            "seed": config.seed,
            "unix_ms": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }));
    }
    let metrics_ref = metrics.as_ref();
    let outcomes: Vec<GameOutcome> = if config.single_thread {
        seeds
            .iter()
            .enumerate()
            .map(|(i, &s)| run_one(s, i, &config.strategies, record_dir, metrics_ref, config.rollout_cfg, &config.net, config.alpha_cfg))
            .collect()
    } else {
        seeds
            .par_iter()
            .enumerate()
            .map(|(i, &s)| run_one(s, i, &config.strategies, record_dir, metrics_ref, config.rollout_cfg, &config.net, config.alpha_cfg))
            .collect()
    };
    let elapsed = start.elapsed();

    let mut wins = [0usize; 4];
    let mut no_winner = 0usize;
    let mut total_turns = 0u64;
    let mut total_steps = 0u64;
    for o in &outcomes {
        if o.winner >= 0 {
            wins[o.winner as usize] += 1;
        } else {
            no_winner += 1;
        }
        total_turns += o.turns as u64;
        total_steps += o.steps;
    }

    let secs = elapsed.as_secs_f64();
    println!("=== catan-sim results ===");
    println!("games:        {}", config.games);
    println!(
        "players:      {}",
        config
            .strategies
            .iter()
            .map(|s| s.name())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "threads:      {}",
        if config.single_thread {
            1
        } else {
            rayon::current_num_threads()
        }
    );
    println!("elapsed:      {:.3}s", secs);
    println!("games/sec:    {:.0}", config.games as f64 / secs);
    println!(
        "avg latency:  {:.3}ms/game",
        secs * 1000.0 / config.games as f64
    );
    println!("steps:        {}", total_steps);
    println!("steps/sec:    {:.0}", total_steps as f64 / secs);
    println!("ns/step:      {:.0}", secs * 1e9 / total_steps as f64);
    println!(
        "avg turns:    {:.1}",
        total_turns as f64 / config.games as f64
    );
    for p in 0..n {
        println!(
            "player {p} ({}): {} wins ({:.1}%)",
            config.strategies[p].name(),
            wins[p],
            wins[p] as f64 * 100.0 / config.games as f64
        );
    }
    if no_winner > 0 {
        println!("no winner (turn cap): {no_winner}");
    }
}
