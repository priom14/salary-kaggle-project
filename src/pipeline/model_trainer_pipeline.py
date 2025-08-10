from src.config.configuration import ConfigurationManger
from src.components.model_trainer import ModelTrainer
from src import logger


STAGE_NAME = "Model Training Stage "


class ModelTrainingPipeline:
    def __init__(self):
        pass
    def initiate_model_training(self):
        config = ConfigurationManger()
        model_training_config = config.get_model_tariner_config()
        model_trainer = ModelTrainer(config = model_training_config)
        model_trainer.train()
        



if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.initiate_model_training()
        logger.info(f">>>>>>>>{STAGE_NAME} completed<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
