import os
import json
import sqlalchemy
import base64
import codecs
import os
import pymssql
import logging
import pathlib

logger = logging.getLogger(__name__)

default_data_directory = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../data'))
default_config_directory = os.getenv('DATA_PASS', default_data_directory)


def get_config(base, config_directory='.'):
    """Retrieve the configuration for the target database/API from json file.

    :param base: Identifier of the target object in the configuration file.
    :param age: Name of the configuration file..

    For security reasons, configuration files are not included in the package.
    The configuration files must be located in happytal/data/ directory.
    """
    config_filename = pathlib.Path(config_directory)/(base+'.json')
    logger.debug('get_config( {} )'.format(str(config_filename)))
    with config_filename.open() as config_file:
        config_data = json.loads(config_file.read())

    return config_data


def get_engine(base, config_directory=default_config_directory):
    """
    Return an sqlalchemy engine.

    :param base: Identifier of the database.
    :param config_directory: Name of the file containing databases parameters.
    """
    config_data = get_config(base, config_directory)
    driver_to_prefix = {
        '{PostgreSQL Unicode}': 'postgresql',
        '{ODBC Driver 13 for SQL Server}': 'sql',
        '{ODBC Driver 17 for SQL Server}': 'sql',
        '{SQLite3 ODBC Driver}': 'sqlite'
    }

    engine = None
    prefix = driver_to_prefix[config_data['driver']]
    if prefix == 'sqlite':
        engine = sqlalchemy.create_engine(
            '{}:///{}'.format(
                driver_to_prefix[config_data['driver']],
                config_data['database']))
    elif prefix == 'sql':
        engine = pymssql.connect(
            host=config_data['server'],
            port=config_data['port'],
            user=config_data['username'],
            password=config_data['password'],
            database=config_data['database'])

    else:
        engine = sqlalchemy.create_engine('{}://{}:{}@{}:{}/{}'.format(
            driver_to_prefix[config_data['driver']],
            config_data['username'],
            config_data['password'],
            config_data['server'],
            int(config_data['port']) if 'port' in config_data else 5432,
            config_data['database']))

    return engine


def get_header(base, config_directory=default_config_directory):
    """
    Return the header to access an API.

    :param base: Identifier of the API.
    :param config_directory: Name of the file containing header parameters.
    """
    config_data = get_config(base, config_directory)

    adress = 'http://' + config_data['server']
    header = {"Authorization": "Basic {user}".format(
        user=base64.b64encode(
            codecs.encode(
                config_data['username'] + ':' + config_data['password']))
        .decode('utf-8'))}
    return adress, header
