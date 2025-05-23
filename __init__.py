from .utils.utils import version

from . import detector, tracker, reid, utils

__all__ = [
    'detector',
    'tracker',
    'reid',
    'utils',
]

__version__ = version()