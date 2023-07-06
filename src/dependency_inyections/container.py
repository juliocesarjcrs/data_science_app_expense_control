from dependency_injector import containers, providers
from utils.utils import Utils
from src.training.model_training import ModelTraining

class Container(containers.DeclarativeContainer):
    utils = providers.Singleton(Utils)
    model_training = providers.Factory(ModelTraining, utils=utils)

