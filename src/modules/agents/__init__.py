from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .rnn_agent_emc import RNNAgentEMC
from .rnn_agent_cds import RNNAgentCDS

REGISTRY = {"rnn": RNNAgent,
            "rnn_ns": RNNNSAgent,
            "rnn_feat": RNNFeatureAgent,
            "rnn_emc": RNNAgentEMC,
            "rnn_cds": RNNAgentCDS}
