from catan_rl.features.board_cnn import BoardCNN, BoardCNNAttn, MLPExtractor
from catan_rl.features.hex_encoder import HexEncoder

EXTRACTORS: dict[str, type] = {
    "boardcnn": BoardCNN,
    "boardcnn_attn": BoardCNNAttn,
    "mlp": MLPExtractor,
    "hex": HexEncoder,
}

__all__ = ["BoardCNN", "BoardCNNAttn", "MLPExtractor", "HexEncoder", "EXTRACTORS"]
