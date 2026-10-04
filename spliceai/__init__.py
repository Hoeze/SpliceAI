import signal
from importlib.metadata import version


signal.signal(signal.SIGINT, lambda x, y: exit(0))

name = 'spliceai'
__version__ = version(name)
