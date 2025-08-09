from src import logger
from src.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.pipeline.data_validation_pipeline import DataValidationTrainingPipeline



# STAGE_NAME = "Data Ingestion Stage"

# try:
#     logger.info(f">>>>> stage {STAGE_NAME} started<<<<<<")
#     data_ingestion = DataIngestionTrainingPipeline()
#     data_ingestion.initiate_data_ingestion()
#     logger.info(f">>>>>>> stage {STAGE_NAME} completed<<<<<<")
    
# except Exception as e:
#     logger.exception(e)
#     raise e    

STAGE_NAME = "Data validation Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.initiate_data_validaiton()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed<<<<<<")
    
except Exception as e:
    logger.exception(e)
    raise e    