import logging

def create_logger(module_name:str):
    logging.basicConfig(encoding='utf-8', level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)-7s %(message)s")

    # Creating a logger
    logger  = logging.getLogger(module_name)

    # Setting up a FileHandler for this logger
    handler  = logging.FileHandler(f'utils/logs/{module_name}.log', encoding='utf-8')
    handler.setLevel(logging.INFO)

    # # Defining the logging format
    formatter  = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-7s %(message)s")
    handler.setFormatter(formatter)

    # Adding the handler to the logger
    logger.addHandler(handler)

    return logger