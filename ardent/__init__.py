from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version(__name__)
    __author__ = 'Manu Stalport'
except PackageNotFoundError:
    pass
