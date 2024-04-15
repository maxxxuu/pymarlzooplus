from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner

REGISTRY = {"q_learner": QLearner,
            "coma_learner": COMALearner,
            "qtran_learner": QTranLearner,
            "actor_critic_learner": ActorCriticLearner,
            "maddpg_learner": MADDPGLearner,
            "ppo_learner": PPOLearner,
            "pac_learner": PACActorCriticLearner,
            "pac_dcg_learner": PACDCGLearner
            }

