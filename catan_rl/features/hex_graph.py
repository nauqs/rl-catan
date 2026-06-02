"""Static tile/node graph for the Tile Transformer (HexFormer).

Builds, once, the token layout + adjacency for a BASE Catan board:

  * 73 tokens = 19 land tiles (token ids 0..18, = tile.id) + 54 nodes
    (token ids 19..72, = 19 + node_id).
  * ``token_cells`` — the flattened cell index (``i * W + j``) in the conv
    feature-map grid (HEIGHT, WIDTH) = the same (21, 11) system used by
    ``get_node_and_edge_maps`` (NOT the divergent ``get_tile_coordinate_map``
    grid). Node cells come straight from ``node_map``; tile cells are the
    rounded centroid of the tile's 6 node cells, keeping everything in one
    coordinate system.
  * ``adj_mask`` — (73, 73) bool, True where attention is allowed:
    self-loops, tile<->its 6 nodes, node<->node (board edges), tile<->tile
    (tiles sharing an edge, i.e. >=2 shared nodes).

The topology is template-fixed for the BASE map (resource/number shuffling
does not change which node belongs to which tile), so one instance suffices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from catanatron.gym.board_tensor_features import get_node_and_edge_maps

N_TILES = 19
N_NODES = 54
N_TOKENS = N_TILES + N_NODES  # 73


@dataclass(frozen=True)
class HexGraph:
    token_cells: np.ndarray   # (73,) int64, flattened i*W+j
    adj_mask: np.ndarray      # (73, 73) bool
    token_type: np.ndarray    # (73,) int64, 0=tile 1=node
    height: int
    width: int


def _base_map():
    from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE
    return CatanMap.from_template(BASE_MAP_TEMPLATE)


def build_hex_graph(height: int = 21, width: int = 11) -> HexGraph:
    node_map, _edge_map = get_node_and_edge_maps()
    cmap = _base_map()

    token_cells = np.zeros(N_TOKENS, dtype=np.int64)
    token_type = np.zeros(N_TOKENS, dtype=np.int64)
    adj = np.eye(N_TOKENS, dtype=bool)  # self-loops

    # Node tokens: ids 0..53 -> token 19+id, cell from node_map.
    node_cell_ij: dict[int, tuple[int, int]] = {}
    for nid in range(N_NODES):
        i, j = node_map[nid]
        node_cell_ij[nid] = (int(i), int(j))
        tok = N_TILES + nid
        token_cells[tok] = int(i) * width + int(j)
        token_type[tok] = 1

    # Tile tokens + incidence + node-node edges + tile-tile adjacency.
    tile_nodes: dict[int, set[int]] = {}
    for _cube, tile in cmap.land_tiles.items():
        tid = int(tile.id)
        nodes = {int(n) for n in tile.nodes.values()}
        tile_nodes[tid] = nodes

        # tile centroid cell (mean of its node cells) in the (H, W) grid
        ijs = np.array([node_cell_ij[n] for n in nodes])
        ci, cj = int(round(ijs[:, 0].mean())), int(round(ijs[:, 1].mean()))
        ci = min(max(ci, 0), height - 1)
        cj = min(max(cj, 0), width - 1)
        token_cells[tid] = ci * width + cj
        token_type[tid] = 0

        # tile <-> its nodes
        for n in nodes:
            adj[tid, N_TILES + n] = True
            adj[N_TILES + n, tid] = True

        # node <-> node along this tile's edges
        for (a, b) in tile.edges.values():
            ta, tb = N_TILES + int(a), N_TILES + int(b)
            adj[ta, tb] = True
            adj[tb, ta] = True

    # tile <-> tile: share an edge (>= 2 common nodes)
    tids = sorted(tile_nodes)
    for x in range(len(tids)):
        for y in range(x + 1, len(tids)):
            tx, ty = tids[x], tids[y]
            if len(tile_nodes[tx] & tile_nodes[ty]) >= 2:
                adj[tx, ty] = True
                adj[ty, tx] = True

    return HexGraph(token_cells, adj, token_type, height, width)
