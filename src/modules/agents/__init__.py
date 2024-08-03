from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .mlp_mat_agent import MLPMATAgent

REGISTRY = {"rnn": RNNAgent,
            "rnn_ns": RNNNSAgent,
            "rnn_feat": RNNFeatureAgent,
            "mlp_mat": MLPMATAgent
}

