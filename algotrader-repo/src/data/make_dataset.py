# -*- coding: utf-8 -*-
import os
import click
import logging
import yaml
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from merge_assets import merge_assets
from load_from_yfinance import load_asset


@click.command()
@click.argument('assets_yaml_filepath', type=click.Path(exists=True), default='assets.yml')
@click.argument('freq', default='1d')
@click.argument('output_filepath', type=click.Path(), default=os.path.join('../../data/interim'))
def main(assets_yaml_filepath, freq, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    
    with open(assets_yaml_filepath, 'r') as file:
        assets_dict = yaml.load(file, Loader=yaml.FullLoader)
    
    logger.info('downloading and stroing assets')

    for key in assets_dict.keys():
        load_asset(assets_dict[key], freq=freq)

    logger.info('making final data set from raw data')

    merge_assets(output_filepath, freq=freq)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
