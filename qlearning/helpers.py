from typing import Mapping
from typing import Type
from typing import Any

import bz2
import pickle
import datetime
import logging
from pathlib import Path

from config import Config


def get_configuration(config: str, configurations: Mapping[str, Type[Config]]) -> Type[Config]:
    try:
        configuration = configurations[config]
    except KeyError:
        raise ValueError('Not a known configuration: {}'.format(config))

    return configuration


def get_logger() -> logging.Logger:
    logger = logging.getLogger('')
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    return logger


def save(obj: Any) -> None:
    class_name = obj.__class__.__name__.lower()

    logging.info(f'Saving {class_name} to disk...')

    persistence_path = Path(obj.persistence_path)
    if not persistence_path.is_dir():
        persistence_path.mkdir()

    pickle_file = Path(f'{class_name}_{datetime.datetime.utcnow()}.pbz2')

    obj.pickle_file = (persistence_path / pickle_file).as_posix()

    handler = bz2.BZ2File(obj.pickle_file, 'w')
    with handler as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    logging.info(f'{class_name.title()} saved to {obj.pickle_file}')


def find_latest_file(path: str) -> str:
    files = Path(path).glob('*.pbz2')
    latest_file = max(files, key=lambda file: file.stat().st_ctime)

    return latest_file.as_posix()


def load_input_file(input_file: str) -> Any:
    if input_file.endswith('pbz2'):
        handler = bz2.open(input_file, 'rb')
    elif input_file.endswith('pkl'):
        handler = open(input_file, 'rb')

    with handler as f:
        data = pickle.load(f)

    return data
