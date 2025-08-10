from src.config.configuration import ConfigurationManger
from src.components.model_evaluation import ModelEvaluation
from src import logger


STAGE_NAME = "Model Evaluation Stage "


class ModelEvaluationPipeline:
    def __init__(self):
        pass
    def initiate_model_evaluation(self):
        config = ConfigurationManger()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config = model_evaluation_config)
        model_evaluation.log_into_mlflow()
        



if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.initiate_model_evaluation()
        logger.info(f">>>>>>>>{STAGE_NAME} completed<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
