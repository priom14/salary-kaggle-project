from src.config.configuration import ConfigurationManger
from src.components.data_transformation import DataTransfroamtion
from src import logger


STAGE_NAME = "Data Transformation Stage "


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    def initiate_data_transformation(self):
        config = ConfigurationManger()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransfroamtion(config = data_transformation_config)
        data_transformation.create_train_test_sets()
        



if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.initiate_data_transformation()
        logger.info(f">>>>>>>>{STAGE_NAME} completed<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
