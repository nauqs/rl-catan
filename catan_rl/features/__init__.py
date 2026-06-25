from catan_rl.features.board_cnn import BoardCNN, BoardCNNAttn, MLPExtractor
from catan_rl.features.hex_encoder import HexEncoder
from catan_rl.features.hexformer import HexFormer
from catan_rl.features.xdim import XdimTrunk

EXTRACTORS: dict[str, type] = {
    "boardcnn": BoardCNN,
    "boardcnn_attn": BoardCNNAttn,
    "mlp": MLPExtractor,
    "hex": HexEncoder,
    "hexformer": HexFormer,
    "xdim": XdimTrunk,
}

__all__ = [
    "BoardCNN", "BoardCNNAttn", "MLPExtractor", "HexEncoder", "HexFormer", "XdimTrunk",
    "EXTRACTORS",
]
