"""Coordinate bijections between the Rust engine board and Catanatron's board.

Both engines model the identical standard 19-hex board; they differ only in
indexing: the engine uses flat ids (tile 0..18 row-by-row, vertex 0..53 in
first-seen order, edge 0..71) while Catanatron uses cube coordinates for tiles
and its own node ids. This module derives the bijection GEOMETRICALLY:

  1. Replicate the engine's exact layout loop (rows 3-4-5-4-3, pointy-top,
     integer half-width units — rust/catan-core/src/board.rs::Topology::build)
     to get every rust tile center and vertex coordinate, then CERTIFY the
     replication against catan_py.topology()'s tile_vertices.
  2. Map Catanatron cube coords into the same integer plane with a fixed
     linear transform, matching tile centers 1:1.
  3. Resolve Catanatron node ids to plane coordinates via each tile's NodeRef
     corners, trying the two vertical orientations and keeping the one that is
     globally consistent (every node gets exactly one coordinate).
  4. Edges follow from endpoint pairs. Every map is validated to be a total
     bijection before being returned — a failure raises, never mis-translates.

Any consistent isomorphism is correct: boards are randomized per game, so the
net reads resources/numbers as features, not absolute positions.
"""
from __future__ import annotations

from functools import lru_cache

ROWS = [3, 4, 5, 4, 3]
# Pointy-top corner offsets clockwise from the top corner (matches board.rs).
CORNERS = [(0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1)]

# NodeRef corner offsets, hypothesis A (y grows south, like the engine) and
# its vertical flip B (Catanatron's +z may point the other way).
_NODE_OFFSETS_A = {
    "NORTH": (0, -2), "NORTHEAST": (1, -1), "SOUTHEAST": (1, 1),
    "SOUTH": (0, 2), "SOUTHWEST": (-1, 1), "NORTHWEST": (-1, -1),
}
_NODE_OFFSETS_B = {k: (dx, -dy) for k, (dx, dy) in _NODE_OFFSETS_A.items()}


def _rust_layout():
    """Replicate Topology::build: tile centers + coord->vertex-id assignment."""
    coord_to_vid: dict[tuple[int, int], int] = {}
    tile_centers: list[tuple[int, int]] = []
    tile_vertices: list[list[int]] = []
    for row, n in enumerate(ROWS):
        y = 3 * (row - 2)
        x_start = -(n - 1)
        for col in range(n):
            cx = x_start + 2 * col
            tile_centers.append((cx, y))
            vids = []
            for dx, dy in CORNERS:
                key = (cx + dx, y + dy)
                if key not in coord_to_vid:
                    coord_to_vid[key] = len(coord_to_vid)
                vids.append(coord_to_vid[key])
            tile_vertices.append(vids)
    assert len(coord_to_vid) == 54 and len(tile_centers) == 19
    return tile_centers, coord_to_vid, tile_vertices


def _cube_to_plane(cube) -> tuple[int, int]:
    """Catanatron cube (x,y,z) -> engine integer plane. Pointy-top axial:
    q = x, r = z; horizontal step 2 half-widths, row step 3 quarter-heights."""
    x, y, z = cube
    return (2 * x + z, 3 * z)


@lru_cache(maxsize=1)
def build_maps():
    """Derive and validate all bijections. Returns a dict:
        tile_c2r: catanatron cube coord -> rust tile id (0..18)
        v_c2r:    catanatron node id    -> rust vertex id (0..53)
        e_c2r:    frozenset{n1,n2}      -> rust edge id (0..71)
      plus the inverses (tile_r2c, v_r2c, e_r2c).
    Requires catanatron and catan_py importable; raises on any inconsistency.
    """
    import catan_py
    from catanatron import Game, Color
    from catanatron.players.weighted_random import WeightedRandomPlayer

    # -- 1. rust layout, certified against the built engine topology --------
    tile_centers, coord_to_vid, tv_py = _rust_layout()
    topo = catan_py.topology()
    assert [list(r) for r in tv_py] == [list(r) for r in topo["tile_vertices"]], \
        "python replica of Topology::build diverges from catan_py.topology()"
    edge_by_pair = {frozenset(ev): eid for eid, ev in enumerate(topo["edge_vertices"])}

    # -- 2. tiles: cube -> plane -> rust tile id ----------------------------
    # (Catanatron's land grid is game-independent; a throwaway game exposes it.)
    game = Game([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)])
    land = game.state.board.map.land_tiles
    center_to_tile = {c: t for t, c in enumerate(tile_centers)}
    tile_c2r: dict = {}
    for cube in land:
        plane = _cube_to_plane(cube)
        if plane not in center_to_tile:
            raise AssertionError(f"cube {cube} -> plane {plane} not a rust center")
    for cube in land:
        tile_c2r[cube] = center_to_tile[_cube_to_plane(cube)]
    assert len(set(tile_c2r.values())) == 19

    # -- 3. nodes: NodeRef corners under the consistent orientation ---------
    def try_offsets(offsets):
        v_c2r: dict = {}
        for cube, tile in land.items():
            cx, cy = _cube_to_plane(cube)
            for ref, node in tile.nodes.items():
                off = offsets[ref.value]
                coord = (cx + off[0], cy + off[1])
                vid = coord_to_vid.get(coord)
                if vid is None:
                    return None
                if v_c2r.setdefault(node, vid) != vid:
                    return None          # same node resolved to two coords
        return v_c2r if len(v_c2r) == 54 and len(set(v_c2r.values())) == 54 else None

    v_c2r = try_offsets(_NODE_OFFSETS_A) or try_offsets(_NODE_OFFSETS_B)
    assert v_c2r is not None, "no consistent NodeRef orientation found"

    # -- 4. edges ------------------------------------------------------------
    e_c2r: dict = {}
    for cube, tile in land.items():
        for _, (n1, n2) in tile.edges.items():
            eid = edge_by_pair.get(frozenset((v_c2r[n1], v_c2r[n2])))
            assert eid is not None, f"edge {(n1, n2)} maps to no rust edge"
            e_c2r[frozenset((n1, n2))] = eid
    assert len(set(e_c2r.values())) == 72, f"edges covered: {len(set(e_c2r.values()))}"

    return {
        "tile_c2r": tile_c2r,
        "v_c2r": v_c2r,
        "e_c2r": e_c2r,
        "tile_r2c": {v: k for k, v in tile_c2r.items()},
        "v_r2c": {v: k for k, v in v_c2r.items()},
        "e_r2c": {v: k for k, v in e_c2r.items()},
    }


if __name__ == "__main__":
    m = build_maps()
    print("bijections OK:",
          f"{len(m['tile_c2r'])} tiles, {len(m['v_c2r'])} nodes, {len(m['e_c2r'])} edge-keys")
