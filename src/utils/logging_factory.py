import logging
from typing import Optional


class LoggerFactory:
    __loggers = {}

    @staticmethod
    def get_logger(name: Optional[str] = None, level = logging.INFO) -> logging.Logger:
        """
        Returns a logger with consistent formatting and optional module name.

        Parameters
        ----------
        name: string (optional)
            Logger name
        level: logging level

        Returns
        -------
        C: Logger instance
        """

        if name is None:
            name = 'default'

        if name not in LoggerFactory.__loggers:
            logger = logging.getLogger(name)
            logger.setLevel(level)

            if not logger.handlers:  # Avoid duplicate handlers
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            LoggerFactory.__loggers[name] = logger

        return LoggerFactory.__loggers[name]