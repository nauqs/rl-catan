from catan_rl.features.board_cnn import BoardCNN, BoardCNNAttn, MLPExtractor

EXTRACTORS: dict[str, type] = {
    "boardcnn": BoardCNN,
    "boardcnn_attn": BoardCNNAttn,
    "mlp": MLPExtractor,
}

__all__ = ["BoardCNN", "BoardCNNAttn", "MLPExtractor", "EXTRACTORS"]
