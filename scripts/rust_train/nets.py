"""Policy/value networks for the Rust-engine PPO trainer (ppo_rust.py).

Two drop-in architectures with the SAME forward signature
`forward(obs[B,1350] float, mask[B,299] bool) -> (logits[B,299], value[B])`:

  - MlpPolicyValueNet:    the proven baseline (2-layer MLP, ~488k params).
  - HexFormerPolicyValueNet: a Tile-Transformer port of our Phase-3/4 spatial
    architecture, adapted to the Rust engine's obs/action layout.

HexFormer rationale: the engine's flat obs is a fixed, documented layout
(rust/catan-env/src/obs.rs, OBS_VERSION=1) that reshapes cleanly into per-board
tokens — 19 tiles, 54 vertices, 72 edges — plus one global context token. The
299-action space (codec.rs, CODEC_VERSION=1) splits into spatial pointer subsets
read directly off those tokens, plus a flat head for non-spatial actions:

    settlement 0..53   <- 54 vertex tokens
    city       54..107 <- same 54 vertex tokens
    road       108..179<- 72 edge tokens
    robber     180..198<- 19 tile tokens
    flat       199..298<- global token (steal/discard/mono/yop/bank/propose/
                          respond/confirm/roll/buy_dev/knight/road_build/end)

Because we train ON the engine, obs token index i and action vertex/edge/tile
index i are the SAME engine index, so the pointer head is aligned by
construction — no coordinate translation (that is only needed for the
Catanatron eval adapter).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# obs.rs block offsets (start, count, stride) — keep in sync with OBS_VERSION.
_TILES, _N_TILES, _TILE_F = 0, 19, 8
_VERTS, _N_VERTS, _VERT_F = 152, 54, 14
_EDGES, _N_EDGES, _EDGE_F = 908, 72, 4
_GLOBAL_START = 1196          # players|self|opp|bank|context|trade
_GLOBAL_F = 1350 - _GLOBAL_START  # 154

# codec.rs spatial action counts (their concatenation order == action-id order).
_N_SETTLE, _N_CITY, _N_ROAD, _N_ROBBER = 54, 54, 72, 19
_N_FLAT = 299 - (_N_SETTLE + _N_CITY + _N_ROAD + _N_ROBBER)  # 100 (ids 199..298)


# --------------------------------------------------------------------------- #
# Board adjacency (from the Rust engine's own indexing) for the graph-masked
# and tiles-only attention variants. Fetched once from catan_py.topology().
# --------------------------------------------------------------------------- #
def _topology():
    import catan_py
    t = catan_py.topology()
    return {k: [list(row) for row in v] for k, v in t.items()}


def build_graph_mask() -> torch.Tensor:
    """[146,146] bool attention mask (True = may attend) for the graph variant.

    Token order: [global(1) | tiles(19) | vertices(54) | edges(72)]. A board
    token attends to its real topological neighbors (share the board graph) +
    itself; the global token is a hub (attends to / attended by everyone) so
    whole-board context still reaches every token in 2 hops. -1 padding skipped.
    """
    topo = _topology()
    G, T, V, E = 1, 19, 54, 72
    N = G + T + V + E
    ti = lambda t: 1 + t          # tile token index
    vi = lambda v: 1 + T + v      # vertex token index
    ei = lambda e: 1 + T + V + e  # edge token index
    m = torch.zeros(N, N, dtype=torch.bool)
    m.fill_diagonal_(True)
    m[0, :] = True                # global hub
    m[:, 0] = True

    def link(a, b):
        m[a, b] = True
        m[b, a] = True

    tile_verts = topo["tile_vertices"]        # [19][6]
    vert_tiles = topo["vertex_tiles"]         # [54][3]
    vert_nbrs = topo["vertex_neighbors"]      # [54][3]
    vert_edges = topo["vertex_edges"]         # [54][3]
    edge_verts = topo["edge_vertices"]        # [72][2]

    for t in range(T):
        vs = [v for v in tile_verts[t] if v >= 0]
        for v in vs:                          # tile <-> its corner vertices
            link(ti(t), vi(v))
        for e in range(E):                    # tile <-> edges fully inside it
            a, b = edge_verts[e]
            if a in vs and b in vs:
                link(ti(t), ei(e))
    # tile <-> tile if they share an edge (>=2 common vertices)
    for t1 in range(T):
        s1 = set(v for v in tile_verts[t1] if v >= 0)
        for t2 in range(t1 + 1, T):
            s2 = set(v for v in tile_verts[t2] if v >= 0)
            if len(s1 & s2) >= 2:
                link(ti(t1), ti(t2))
    for v in range(V):
        for n in vert_nbrs[v]:                # vertex <-> neighbor vertices
            if n >= 0:
                link(vi(v), vi(n))
        for e in vert_edges[v]:               # vertex <-> incident edges
            if e >= 0:
                link(vi(v), ei(e))
        for t in vert_tiles[v]:               # vertex <-> its tiles
            if t >= 0:
                link(vi(v), ti(t))
    for e in range(E):
        a, b = edge_verts[e]                  # edge <-> its 2 endpoints
        link(ei(e), vi(a))
        link(ei(e), vi(b))
        for e2 in range(e + 1, E):            # edge <-> edges sharing a vertex
            if set(edge_verts[e]) & set(edge_verts[e2]):
                link(ei(e), ei(e2))
    return m


def build_tile_mask() -> torch.Tensor:
    """[20,20] bool attention mask for the tiles-only GRAPH variant.

    Token order: [global(1) | tiles(19)]. A tile attends to hex-adjacent tiles
    (share an edge = >=2 common vertices) + itself; global is a hub. Turns the
    tiles trunk from all-to-all into board-structured (Charlesworth used tiles
    all-to-all; this tests whether masking even 19 tokens to adjacency helps).
    """
    topo = _topology()
    T = 19
    N = 1 + T
    m = torch.zeros(N, N, dtype=torch.bool)
    m.fill_diagonal_(True)
    m[0, :] = True
    m[:, 0] = True
    tv = topo["tile_vertices"]
    for a in range(T):
        sa = set(v for v in tv[a] if v >= 0)
        for b in range(a + 1, T):
            if len(sa & set(v for v in tv[b] if v >= 0)) >= 2:
                m[1 + a, 1 + b] = True
                m[1 + b, 1 + a] = True
    return m


def _pad_index(rows, width):
    """[K][<=width] ragged (-1 padded) -> (idx[K,width] long, mask[K,width] float)."""
    K = len(rows)
    idx = torch.zeros(K, width, dtype=torch.long)
    msk = torch.zeros(K, width)
    for i, row in enumerate(rows):
        for j, val in enumerate(row[:width]):
            if val >= 0:
                idx[i, j] = val
                msk[i, j] = 1.0
    return idx, msk


class MlpPolicyValueNet(nn.Module):
    """Baseline: identical to the original ppo_rust.py PolicyValueNet."""

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, num_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        h = self.trunk(obs)
        logits = self.policy(h).masked_fill(~mask, float("-inf"))
        return logits, self.value(h).squeeze(-1)


class _SDPABlock(nn.Module):
    """Pre-norm transformer block using fused scaled_dot_product_attention.

    Replaces nn.TransformerEncoderLayer: ~15% faster on MPS, fp16-ready, and no
    enable_nested_tensor warning. All-to-all (no attention mask) — the board is
    tiny (146 tokens) so global attention is cheap and lets every token see every
    other (Charlesworth's tile encoder was also all-to-all).
    """

    def __init__(self, d: int, heads: int, ff_mult: int = 2):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * ff_mult), nn.GELU(), nn.Linear(d * ff_mult, d),
        )

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape
        h = self.n1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # attn_mask: [N,N] bool (True = attend) or None (all-to-all).
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        a = a.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(a)
        x = x + self.ff(self.n2(x))
        return x


class HexFormerPolicyValueNet(nn.Module):
    """Tile-Transformer trunk + spatial pointer policy head.

    Tokens: [global] + 19 tiles + 54 vertices + 72 edges (146 tokens). Per-type
    input projections, learnable per-slot identity embeddings, an all-to-all
    transformer encoder (board tokens attend freely — Charlesworth's tile encoder
    was also all-to-all; graph masking can be layered on later), then pointer
    heads on the board tokens and a flat head + value head on the global token.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128,
                 n_layers: int = 3, n_heads: int = 4, ff_mult: int = 2,
                 dropout: float = 0.0, attn: str = "full"):
        super().__init__()
        assert obs_dim == 1350 and num_actions == 299, \
            "HexFormer is wired to OBS_VERSION=1 / CODEC_VERSION=1 (1350/299)"
        assert attn in ("full", "graph"), attn
        self.attn = attn
        d = hidden
        self.d_model = d
        # Graph variant: attend only along the real board graph (+ global hub).
        if attn == "graph":
            self.register_buffer("attn_mask", build_graph_mask(), persistent=False)
        else:
            self.attn_mask = None

        self.tile_proj = nn.Linear(_TILE_F, d)
        self.vertex_proj = nn.Linear(_VERT_F, d)
        self.edge_proj = nn.Linear(_EDGE_F, d)
        self.global_proj = nn.Sequential(
            nn.Linear(_GLOBAL_F, d), nn.ReLU(), nn.Linear(d, d),
        )
        # Learnable identity embeddings so token i is distinguishable.
        self.tile_id = nn.Parameter(torch.zeros(_N_TILES, d))
        self.vertex_id = nn.Parameter(torch.zeros(_N_VERTS, d))
        self.edge_id = nn.Parameter(torch.zeros(_N_EDGES, d))
        self.global_id = nn.Parameter(torch.zeros(1, d))

        self.encoder = nn.ModuleList(
            [_SDPABlock(d, n_heads, ff_mult) for _ in range(n_layers)]
        )

        self.settle_head = nn.Linear(d, 1)
        self.city_head = nn.Linear(d, 1)
        self.road_head = nn.Linear(d, 1)
        self.robber_head = nn.Linear(d, 1)
        self.flat_head = nn.Linear(d, _N_FLAT)
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        tiles = obs[:, _TILES:_TILES + _N_TILES * _TILE_F].reshape(B, _N_TILES, _TILE_F)
        verts = obs[:, _VERTS:_VERTS + _N_VERTS * _VERT_F].reshape(B, _N_VERTS, _VERT_F)
        edges = obs[:, _EDGES:_EDGES + _N_EDGES * _EDGE_F].reshape(B, _N_EDGES, _EDGE_F)
        glob = obs[:, _GLOBAL_START:]

        t = self.tile_proj(tiles) + self.tile_id
        v = self.vertex_proj(verts) + self.vertex_id
        e = self.edge_proj(edges) + self.edge_id
        g = (self.global_proj(glob) + self.global_id).unsqueeze(1)

        x = torch.cat([g, t, v, e], dim=1)  # [B, 146, d]
        for blk in self.encoder:
            x = blk(x, attn_mask=self.attn_mask)

        g_out = x[:, 0]
        i = 1
        t_out = x[:, i:i + _N_TILES]; i += _N_TILES
        v_out = x[:, i:i + _N_VERTS]; i += _N_VERTS
        e_out = x[:, i:i + _N_EDGES]

        settle = self.settle_head(v_out).squeeze(-1)   # [B,54]
        city = self.city_head(v_out).squeeze(-1)       # [B,54]
        road = self.road_head(e_out).squeeze(-1)       # [B,72]
        robber = self.robber_head(t_out).squeeze(-1)   # [B,19]
        flat = self.flat_head(g_out)                   # [B,100]
        logits = torch.cat([settle, city, road, robber, flat], dim=1)  # [B,299]
        logits = logits.masked_fill(~mask, float("-inf"))
        value = self.value_head(g_out).squeeze(-1)
        return logits, value


class TilesHexFormer(nn.Module):
    """Charlesworth-faithful: attention over TILES ONLY (+ a global hub token),
    then SCATTER to vertex/edge representations for the spatial pointer heads.

    The transformer mixes only the 19 tile tokens + 1 global token (all-to-all,
    like Charlesworth's tile encoder — small, focused, low-noise). Vertices and
    edges are NOT attended: each vertex rep is formed from its (<=3) adjacent
    tile encodings + its own local obs features; each edge rep from its 2
    endpoint vertex reps + its local features. Far cheaper than the 146-token
    trunk, and matches how a player values a spot (its adjacent hexes).
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128,
                 n_layers: int = 3, n_heads: int = 4, ff_mult: int = 2,
                 attn: str = "full"):
        super().__init__()
        assert obs_dim == 1350 and num_actions == 299
        assert attn in ("full", "graph"), attn
        self.attn = attn
        if attn == "graph":
            self.register_buffer("attn_mask", build_tile_mask(), persistent=False)
        else:
            self.attn_mask = None
        d = hidden
        self.d_model = d
        self.tile_proj = nn.Linear(_TILE_F, d)
        self.global_proj = nn.Sequential(nn.Linear(_GLOBAL_F, d), nn.ReLU(), nn.Linear(d, d))
        self.tile_id = nn.Parameter(torch.zeros(_N_TILES, d))
        self.global_id = nn.Parameter(torch.zeros(1, d))
        self.encoder = nn.ModuleList([_SDPABlock(d, n_heads, ff_mult) for _ in range(n_layers)])

        # scatter indices (static): vertex -> its tiles; edge -> its vertices.
        topo = _topology()
        vt_idx, vt_msk = _pad_index(topo["vertex_tiles"], 3)
        ev_idx, ev_msk = _pad_index(topo["edge_vertices"], 2)
        self.register_buffer("vt_idx", vt_idx, persistent=False)
        self.register_buffer("vt_msk", vt_msk, persistent=False)
        self.register_buffer("ev_idx", ev_idx, persistent=False)
        self.register_buffer("ev_msk", ev_msk, persistent=False)

        # form vertex/edge reps: [pooled-neighbor d | local feats] -> d
        self.vertex_mlp = nn.Sequential(nn.Linear(d + _VERT_F, d), nn.GELU(), nn.Linear(d, d))
        self.edge_mlp = nn.Sequential(nn.Linear(d + _EDGE_F, d), nn.GELU(), nn.Linear(d, d))

        self.settle_head = nn.Linear(d, 1)
        self.city_head = nn.Linear(d, 1)
        self.road_head = nn.Linear(d, 1)
        self.robber_head = nn.Linear(d, 1)
        self.flat_head = nn.Linear(d, _N_FLAT)
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def _masked_mean(self, gathered, msk):
        # gathered [B,K,W,d], msk [K,W] -> [B,K,d]
        w = msk.unsqueeze(0).unsqueeze(-1)            # [1,K,W,1]
        s = (gathered * w).sum(2)
        n = w.sum(2).clamp(min=1.0)
        return s / n

    def forward(self, obs, mask):
        B = obs.shape[0]
        tiles = obs[:, _TILES:_TILES + _N_TILES * _TILE_F].reshape(B, _N_TILES, _TILE_F)
        verts_f = obs[:, _VERTS:_VERTS + _N_VERTS * _VERT_F].reshape(B, _N_VERTS, _VERT_F)
        edges_f = obs[:, _EDGES:_EDGES + _N_EDGES * _EDGE_F].reshape(B, _N_EDGES, _EDGE_F)
        glob = obs[:, _GLOBAL_START:]

        t = self.tile_proj(tiles) + self.tile_id
        g = (self.global_proj(glob) + self.global_id).unsqueeze(1)
        x = torch.cat([g, t], dim=1)                  # [B, 20, d]
        for blk in self.encoder:
            x = blk(x, attn_mask=self.attn_mask)
        g_out = x[:, 0]
        t_out = x[:, 1:]                              # [B,19,d]

        # vertex rep = masked-mean of adjacent tile encodings + local feats
        vt = t_out[:, self.vt_idx]                    # [B,54,3,d]
        v_pool = self._masked_mean(vt, self.vt_msk)   # [B,54,d]
        v_out = self.vertex_mlp(torch.cat([v_pool, verts_f], dim=-1))
        # edge rep = mean of endpoint vertex reps + local feats
        ev = v_out[:, self.ev_idx]                    # [B,72,2,d]
        e_pool = self._masked_mean(ev, self.ev_msk)   # [B,72,d]
        e_out = self.edge_mlp(torch.cat([e_pool, edges_f], dim=-1))

        settle = self.settle_head(v_out).squeeze(-1)
        city = self.city_head(v_out).squeeze(-1)
        road = self.road_head(e_out).squeeze(-1)
        robber = self.robber_head(t_out).squeeze(-1)
        flat = self.flat_head(g_out)
        logits = torch.cat([settle, city, road, robber, flat], dim=1)
        logits = logits.masked_fill(~mask, float("-inf"))
        value = self.value_head(g_out).squeeze(-1)
        return logits, value


def build_net(arch: str, obs_dim: int, num_actions: int, hidden: int, **kw):
    if arch == "mlp":
        return MlpPolicyValueNet(obs_dim, num_actions, hidden)
    if arch == "hexformer":
        return HexFormerPolicyValueNet(
            obs_dim, num_actions, hidden,
            n_layers=kw.get("layers", 3), n_heads=kw.get("heads", 4),
            attn=kw.get("attn", "full"),
        )
    if arch == "tiles":
        return TilesHexFormer(
            obs_dim, num_actions, hidden,
            n_layers=kw.get("layers", 3), n_heads=kw.get("heads", 4),
            attn=kw.get("attn", "full"),
        )
    raise ValueError(f"unknown arch {arch!r}")
