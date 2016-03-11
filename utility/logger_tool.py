
import logging
import sys

class Logger:
    def __init__(self, **kwargs):
        logging.basicConfig(**kwargs)

        root = logging.getLogger()
        ch = logging.StreamHandler(sys.stdout)
        root.addHandler(ch)
        return