"""Catanatron `Player` that runs a Rust-engine-trained policy net.

Phase 10 trains on the vendored Rust engine (`catan_py`); this module is the
bridge to the project's north star: it loads a Rust-engine checkpoint and plays
it inside Catanatron, so Rust-trained agents can be evaluated against the real
`AlphaBetaPlayer(n=2)` ladder with no Rust dependency at eval time.

Three layers:
  1. NET GLUE: reconstruct the net, validate codec/obs versions, masked forward.
  2. COORDINATE MAPS: geometric bijection Rust<->Catanatron (adapter_maps.py,
     fully self-validating).
  3. TRANSLATION (this file): Catanatron state -> the frozen 1350-float obs
     (rust/catan-env/src/obs.rs, OBS_VERSION=1) and engine action ids (0..298,
     codec.rs CODEC_VERSION=1) <-> Catanatron actions.

Known, deliberate approximations (all context-level, none structural):
  - dice value not exposed by Catanatron state -> obs slot 0 (production is
    already resolved when we act, so this is low-information);
  - engine's pre_roll/must_roll collapsed to pre_roll; trade phases unused
    (AB2 never proposes trades; the trade-offer block is zeroed);
  - MOVE_ROBBER couples tile+victim in Catanatron but is two engine steps;
    the victim is chosen with the net's STEAL logits from the same forward.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Frozen codec constants — mirror rust/catan-env/src/{obs.rs,codec.rs}.
# --------------------------------------------------------------------------- #
OBS_DIM = 1350
NUM_ACTIONS = 299
CODEC_VERSION = 1
OBS_VERSION = 1

# obs.rs block offsets
TILES, TILE_STRIDE = 0, 8
VERTICES, VERTEX_STRIDE = 152, 14
EDGES, EDGE_STRIDE = 908, 4
PLAYERS, PLAYER_STRIDE = 1196, 17
SELF_PRIVATE = 1264
OPP_PRIVATE, OPP_STRIDE = 1279, 10
BANK = 1309
CONTEXT = 1314
TRADE = 1334

# codec.rs action-id bases
A_SETTLE, A_CITY, A_ROAD, A_ROBBER = 0, 54, 108, 180
A_STEAL, A_STEAL_SKIP, A_DISCARD, A_MONOPOLY, A_YOP = 199, 202, 203, 208, 213
A_BANK, A_ROLL, A_BUY_DEV, A_KNIGHT, A_ROADBUILD, A_END = 228, 294, 295, 296, 297, 298

# Rust resource order (state.rs/eval.rs): wheat, sheep, wood, brick, ore.
CATA_TO_RUST_RES = {"WHEAT": 0, "SHEEP": 1, "WOOD": 2, "BRICK": 3, "ORE": 4}
# Rust dev order (state.rs): knight, victory_point, road_building, yop, monopoly.
DEV_KEYS = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
# Dice-sum probability (out of 36), scaled like obs.rs (peak 5/36 * 7.2 = 1.0).
_NUM_PROB = {n: (6 - abs(7 - n)) / 36.0 for n in range(2, 13) if n != 7}


def _yop_index(r1: int, r2: int) -> int:
    return r1 * (11 - r1) // 2 + (r2 - r1)


def _pair_index(give: int, recv: int) -> int:
    return give * 4 + (recv if recv < give else recv - 1)


class PolicyValueNet(nn.Module):
    """The baseline MLP (kept for loading MLP checkpoints)."""

    def __init__(self, obs_dim: int, num_actions: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, num_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        h = self.trunk(obs)
        logits = self.policy(h)
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits, self.value(h).squeeze(-1)


def load_rust_net(path: str | Path, device: str = "cpu",
                  arch: str = "mlp", **net_kw) -> nn.Module:
    """Load a Phase-10 checkpoint. arch='mlp' uses the local PolicyValueNet;
    other arches are built via scripts/rust_train/nets.build_net (hexformer,
    tiles — pass hidden/layers/heads/attn through net_kw)."""
    ck = torch.load(str(path), map_location=device, weights_only=False)
    if ck.get("codec_version") != CODEC_VERSION:
        raise ValueError(f"codec_version {ck.get('codec_version')} != {CODEC_VERSION}")
    if ck.get("obs_version") != OBS_VERSION:
        raise ValueError(f"obs_version {ck.get('obs_version')} != {OBS_VERSION}")
    if arch == "mlp":
        hidden = ck["model_state"]["trunk.0.weight"].shape[0]
        net = PolicyValueNet(ck.get("obs_dim", OBS_DIM),
                             ck.get("num_actions", NUM_ACTIONS), hidden)
    else:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                               / "scripts" / "rust_train"))
        from nets import build_net
        net = build_net(arch, OBS_DIM, NUM_ACTIONS, net_kw.pop("hidden", 128),
                        **net_kw)
    net.load_state_dict(ck["model_state"])
    net.to(device).eval()
    return net


# --------------------------------------------------------------------------- #
# State translation
# --------------------------------------------------------------------------- #
def catanatron_state_to_obs(game, color, maps, vps_to_win: int = 10) -> np.ndarray:
    """Mirror encode_obs() (obs.rs) from a Catanatron game, seat-relative."""
    import catanatron.state_functions as sf

    s = game.state
    board = s.board
    colors = list(s.colors)
    n = len(colors)
    seat = colors.index(color)
    rel_of = lambda other: (other - seat) % n
    ps = s.player_state
    pidx = {c: i for i, c in enumerate(colors)}
    out = np.zeros(OBS_DIM, dtype=np.float32)

    tile_c2r, v_c2r, e_c2r = maps["tile_c2r"], maps["v_c2r"], maps["e_c2r"]

    # ---- tiles ----------------------------------------------------------
    robber_t = tile_c2r[board.robber_coordinate]
    for cube, tile in board.map.land_tiles.items():
        t = tile_c2r[cube]
        base = TILES + t * TILE_STRIDE
        res = tile.resource
        out[base + (5 if res is None else CATA_TO_RUST_RES[res])] = 1.0
        if tile.number:
            out[base + 6] = _NUM_PROB.get(tile.number, 0.0) * 7.2
        if t == robber_t:
            out[base + 7] = 1.0

    # ---- vertices: buildings + static ports ------------------------------
    port_nodes = board.map.port_nodes            # {None|'RES': set(node)}
    for node, (bcolor, btype) in board.buildings.items():
        v = v_c2r[node]
        rel = rel_of(pidx[bcolor])
        out[VERTICES + v * VERTEX_STRIDE + rel * 2 + (1 if btype == "CITY" else 0)] = 1.0
    for pres, nodes in port_nodes.items():
        pt = 0 if pres is None else CATA_TO_RUST_RES[pres] + 1
        for node in nodes:
            out[VERTICES + v_c2r[node] * VERTEX_STRIDE + 8 + pt] = 1.0

    # ---- edges: roads -----------------------------------------------------
    seen = set()
    for edge, rcolor in board.roads.items():
        key = frozenset(edge)
        if key in seen:
            continue
        seen.add(key)
        out[EDGES + e_c2r[key] * EDGE_STRIDE + rel_of(pidx[rcolor])] = 1.0

    # ---- players: public block per relative seat --------------------------
    target = float(vps_to_win)
    my_nodes_by_color = {c: set() for c in colors}
    for node, (bcolor, _) in board.buildings.items():
        my_nodes_by_color[bcolor].add(node)
    for rel in range(n):
        p = colors[(seat + rel) % n]
        i = pidx[p]
        base = PLAYERS + rel * PLAYER_STRIDE
        out[base + 0] = sf.player_num_resource_cards(s, p) / 19.0
        out[base + 1] = sf.player_num_dev_cards(s, p) / 25.0
        out[base + 2] = ps[f"P{i}_PLAYED_KNIGHT"] / 14.0
        out[base + 3] = sf.get_visible_victory_points(s, p) / target
        out[base + 4] = ps[f"P{i}_SETTLEMENTS_AVAILABLE"] / 5.0
        out[base + 5] = ps[f"P{i}_CITIES_AVAILABLE"] / 4.0
        out[base + 6] = ps[f"P{i}_ROADS_AVAILABLE"] / 15.0
        out[base + 7] = ps[f"P{i}_LONGEST_ROAD_LENGTH"] / 15.0
        out[base + 8] = float(ps[f"P{i}_HAS_ROAD"])
        out[base + 9] = float(ps[f"P{i}_HAS_ARMY"])
        nodes = my_nodes_by_color[p]
        out[base + 10] = float(bool(nodes & port_nodes.get(None, set())))
        for res, r in CATA_TO_RUST_RES.items():
            out[base + 11 + r] = float(bool(nodes & port_nodes.get(res, set())))
        out[base + 16] = float(s.current_color() == p)

    # ---- self private ------------------------------------------------------
    me = pidx[color]
    hand = sf.get_player_freqdeck(s, color)      # catanatron order
    for res, r in CATA_TO_RUST_RES.items():
        out[SELF_PRIVATE + r] = ps[f"P{me}_{res}_IN_HAND"] / 19.0
    for d, key in enumerate(DEV_KEYS):
        in_hand = ps.get(f"P{me}_{key}_IN_HAND", 0)
        out[SELF_PRIVATE + 5 + d] = in_hand / 5.0
        at_start = ps.get(f"P{me}_{key}_OWNED_AT_START", None)
        bought = max(0, in_hand - at_start) if at_start is not None else 0
        out[SELF_PRIVATE + 10 + d] = bought / 2.0
    _ = hand  # (freqdeck kept for clarity; per-key reads above are canonical)

    # ---- opponents private (perfect info) ----------------------------------
    for rel in range(1, n):
        p = colors[(seat + rel) % n]
        i = pidx[p]
        base = OPP_PRIVATE + (rel - 1) * OPP_STRIDE
        for res, r in CATA_TO_RUST_RES.items():
            out[base + r] = ps[f"P{i}_{res}_IN_HAND"] / 19.0
        for d, key in enumerate(DEV_KEYS):
            out[base + 5 + d] = ps.get(f"P{i}_{key}_IN_HAND", 0) / 5.0

    # ---- bank ---------------------------------------------------------------
    freq = s.resource_freqdeck                   # [wood, brick, sheep, wheat, ore]
    cata_order = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    for ci, res in enumerate(cata_order):
        out[BANK + CATA_TO_RUST_RES[res]] = freq[ci] / 19.0

    # ---- context --------------------------------------------------------------
    prompt = s.current_prompt.value if s.current_prompt else ""
    # Engine semantics are TURN-OWNER-relative for has_rolled/dice (s.has_rolled,
    # s.dice_roll are per-turn state), not actor-relative — matters when we act
    # off-turn (discards, trade responses).
    owner_i = s.current_turn_index
    owner_rolled = bool(ps.get(f"P{owner_i}_HAS_ROLLED", False))
    dice = 0
    if owner_rolled:
        for rec in reversed(s.action_records):
            at = rec.action.action_type.value
            if at == "ROLL" and rec.result is not None:
                dice = rec.result[0] + rec.result[1]
                break
            if at == "END_TURN":
                break
    if s.is_initial_build_phase:
        n_settles = sum(1 for _, (c2, b) in board.buildings.items() if b == "SETTLEMENT")
        out[CONTEXT + (0 if n_settles < n else 1)] = 1.0
    elif game.winning_color() is not None:
        out[CONTEXT + 3] = 1.0
    else:
        out[CONTEXT + 2] = 1.0
        tp = 5                                    # main
        if prompt == "PLAY_TURN" and not owner_rolled:
            tp = 0                                # pre_roll (no ROLL prompt exists)
        elif prompt == "DISCARD":
            tp = 2
        elif prompt == "MOVE_ROBBER":
            tp = 3
        elif s.is_road_building or s.free_roads_available > 0:
            tp = 6
        elif prompt == "DECIDE_TRADE":
            tp = 7
        out[CONTEXT + 4 + tp] = 1.0
    out[CONTEXT + 13] = dice / 12.0
    out[CONTEXT + 14] = float(owner_rolled)
    out[CONTEXT + 15] = s.num_turns / 1000.0
    out[CONTEXT + 16] = target / 10.0
    out[CONTEXT + 17] = (s.free_roads_available or 0) / 2.0
    out[CONTEXT + 18] = 1.0                       # trades left (never proposes)
    if prompt == "DISCARD":
        out[CONTEXT + 19] = (sf.player_num_resource_cards(s, color) // 2) / 10.0
    # trade-offer block stays zero
    return out


def encode_action(action, maps) -> int | None:
    """Catanatron Action -> engine id, or None if unmapped (P2P trade etc.)."""
    t = action.action_type.value
    v = action.value
    if t in ("BUILD_SETTLEMENT",):
        return A_SETTLE + maps["v_c2r"][v]
    if t == "BUILD_CITY":
        return A_CITY + maps["v_c2r"][v]
    if t == "BUILD_ROAD":
        return A_ROAD + maps["e_c2r"][frozenset(v)]
    if t == "MOVE_ROBBER":
        return A_ROBBER + maps["tile_c2r"][v[0]]
    if t == "DISCARD_RESOURCE":
        return A_DISCARD + CATA_TO_RUST_RES[v]
    if t == "PLAY_MONOPOLY":
        return A_MONOPOLY + CATA_TO_RUST_RES[v]
    if t == "PLAY_YEAR_OF_PLENTY":
        rs = sorted(CATA_TO_RUST_RES[r] for r in (v if isinstance(v, (list, tuple)) else [v]))
        r1, r2 = (rs[0], rs[-1]) if len(rs) > 1 else (rs[0], rs[0])
        return A_YOP + _yop_index(r1, r2)
    if t == "MARITIME_TRADE":
        give = CATA_TO_RUST_RES[v[0]]
        recv = CATA_TO_RUST_RES[v[4]]
        return A_BANK + _pair_index(give, recv)
    if t == "ROLL":
        return A_ROLL
    if t == "BUY_DEVELOPMENT_CARD":
        return A_BUY_DEV
    if t == "PLAY_KNIGHT_CARD":
        return A_KNIGHT
    if t == "PLAY_ROAD_BUILDING":
        return A_ROADBUILD
    if t == "END_TURN":
        return A_END
    return None


try:
    from catanatron import Player

    class RustNetCatanatronPlayer(Player):
        """Play a Rust-engine-trained net inside Catanatron (greedy by default).

            p = RustNetCatanatronPlayer(Color.BLUE, ckpt, arch="hexformer",
                                        hidden=128, layers=3, attn="graph")
        """

        def __init__(self, color, model_path: str | Path, device: str = "cpu",
                     deterministic: bool = True, arch: str = "mlp",
                     vps_to_win: int = 10, **net_kw):
            super().__init__(color)
            from catan_rl.eval.adapter_maps import build_maps
            self.maps = build_maps()
            self.net = load_rust_net(model_path, device=device, arch=arch, **net_kw)
            self.device = device
            self.deterministic = deterministic
            self.vps_to_win = vps_to_win

        def decide(self, game, playable_actions):
            if len(playable_actions) == 1:
                return playable_actions[0]
            by_id: dict[int, list] = {}
            for a in playable_actions:
                aid = encode_action(a, self.maps)
                if aid is not None:
                    by_id.setdefault(aid, []).append(a)
            if not by_id:
                return playable_actions[0]

            obs = catanatron_state_to_obs(game, self.color, self.maps, self.vps_to_win)
            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            mask[list(by_id)] = True
            with torch.no_grad():
                o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                m = torch.as_tensor(mask, device=self.device).unsqueeze(0)
                logits, _ = self.net(o, m)
                logits = logits[0]
                if self.deterministic:
                    aid = int(torch.argmax(logits).item())
                else:
                    aid = int(torch.distributions.Categorical(logits=logits).sample())

            cands = by_id[aid]
            if len(cands) == 1:
                return cands[0]
            if A_ROBBER <= aid < A_ROBBER + 19:
                # several victims share the tile: rank by the STEAL logits
                colors = list(game.state.colors)
                seat = colors.index(self.color)
                best, best_score = cands[0], -1e30
                for a in cands:
                    victim = a.value[1]
                    if victim is None:
                        score = float(logits[A_STEAL_SKIP])
                    else:
                        rel = (colors.index(victim) - seat) % len(colors)
                        score = float(logits[A_STEAL + rel - 1]) if rel >= 1 else -1e30
                    if score > best_score:
                        best, best_score = a, score
                return best
            if A_BANK <= aid < A_BANK + 20:
                # same give/recv at several ratios: take the cheapest give
                return min(cands, key=lambda a: sum(x is not None for x in a.value[:4]))
            return cands[0]

except ImportError:  # catanatron not installed; net glue still usable
    RustNetCatanatronPlayer = None  # type: ignore
