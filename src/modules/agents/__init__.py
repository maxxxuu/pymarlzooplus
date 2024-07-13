from .rnn_agent import RNNAgent
from .rnn_emc_agent import RNNEMCAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent

REGISTRY = {"rnn": RNNAgent,
            "rnn_ns": RNNNSAgent,
            "rnn_feat": RNNFeatureAgent,
            "rnn_emc": RNNEMCAgent}
