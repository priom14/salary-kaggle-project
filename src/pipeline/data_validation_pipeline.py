from src.config.configuration import ConfigurationManger
from src.components.data_validation import DataValidation
from src import logger


STAGE_NAME = "Data validation Stage "


class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    def initiate_data_validaiton(self):
        config = ConfigurationManger()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config = data_validation_config)
        data_validation.validate_all_columns()
        



if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.initiate_data_validation()
        logger.info(f">>>>>>>>{STAGE_NAME} completed<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
