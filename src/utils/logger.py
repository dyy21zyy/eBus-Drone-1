import logging
def get_logger(name="ebus_drone_rl"):
 logging.basicConfig(level=logging.INFO); return logging.getLogger(name)
