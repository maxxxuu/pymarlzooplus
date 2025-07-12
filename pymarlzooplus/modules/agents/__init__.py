from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .mlp_mat_agent import MLPMATAgent
from .rnn_agent_happo import RNNAgentHAPPO
from .rnn_agent_emc import RNNAgentEMC
from .rnn_agent_cds import RNNAgentCDS
from .rnn_cent_pe_agent import RNNCentPEAgent
from .rnn_cent_agent import RNNCentAgent
from .rnn_cent_pet_agent import RNNCentPETAgent
from .rnn_cent_pedt_agent import RNNCentPEDTAgent

REGISTRY = {"rnn": RNNAgent,
            "rnn_ns": RNNNSAgent,
            "mlp_mat": MLPMATAgent,
            "rnn_happo": RNNAgentHAPPO,
            "rnn_emc": RNNAgentEMC,
            "rnn_cds": RNNAgentCDS,
            "rnn_cent_pe": RNNCentPEAgent,
            "rnn_cent": RNNCentAgent,
            "rnn_cent_pet": RNNCentPETAgent,
            "rnn_cent_pedt": RNNCentPEDTAgent,
            }


