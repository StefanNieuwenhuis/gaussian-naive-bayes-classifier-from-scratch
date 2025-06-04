from data.base import Transformer
from data.data_splitter import DataSplitter
from data.label_preprocessor import LabelPreProcessor
from data.load_data import IrisDataDownloader
from data.outlier_remover import OutlierRemover
from data.preprocessing_pipeline import PreProcessingPipeline
from data.save_data import SaveData
from utils.logging_factory import LoggerFactory

logger = LoggerFactory.get_logger("PreProcessingPipeline")

def main():
    logger.info("Starting preprocessing pipeline")

    # Step 1: Download and Ingest data
    logger.info("Starting downloading Iris Dataset")
    downloader = IrisDataDownloader()

    try:
        downloader.download()
    except FileExistsError:
        print("Data already downloaded")
    logger.info("Finished downloading Iris Dataset")

    logger.info("Starting data ingestion")
    X, y = downloader.load()
    logger.info("Finished data ingestion")

    # Step 2: Clean and Pre-process
    logger.info("Starting preprocessing data")
    steps: list[Transformer] = [OutlierRemover(),LabelPreProcessor(), ]
    pp_pipeline = PreProcessingPipeline(steps)
    X_clean, y_clean = pp_pipeline.fit_transform(X, y)
    logger.info("Finished preprocessing data")

    # Step 3: Train/Test splitting
    logger.info("Starting train/test splitting")
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.split(X_clean, y_clean)
    logger.info("Finished train/test splitting")

    # Step 4: Save
    logger.info("Starting processed data saving")
    data_saver = SaveData()
    data_saver.save_all(X_train, X_test, y_train, y_test)
    logger.info("Finished processed data saving")


if __name__ == "__main__":
    main()