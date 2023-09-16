################### - Imports - #############################################################
import logging

##################################
############################################################################################


################### - Main Execution - #############################################################
def get_logger(filename):
    # logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    #ch = logging.StreamHandler()
    ch = logging.FileHandler(f'{filename}.log')
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger
