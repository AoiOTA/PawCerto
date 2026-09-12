"""Original Go1/ARX5 RoboDuet automatic method components."""
from .config import SOURCE_REVISION, default_config
from .policy import ArmActorCritic, DogActorCritic, build_models, coordinated_actions
from .observations import RoboDuetObservations
from .controller import RoboDuetController, plan

__all__ = ['SOURCE_REVISION', 'default_config', 'ArmActorCritic', 'DogActorCritic',
           'build_models', 'coordinated_actions', 'RoboDuetObservations', 'RoboDuetController', 'plan']
