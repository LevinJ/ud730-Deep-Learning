import logging
import sys
import logger_tool


_=logger_tool.Logger(filename='example.log',filemode='w',level=logging.DEBUG)

# logging.basicConfig(filename='example.log',filemode='w',level=logging.DEBUG)

# root = logging.getLogger()
# ch = logging.StreamHandler(sys.stdout)
# # ch.setLevel(logging.DEBUG)
# # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # ch.setFormatter(formatter)
# root.addHandler(ch)


logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
logging.warning('%s before you %s', 'Look', 'leap!')