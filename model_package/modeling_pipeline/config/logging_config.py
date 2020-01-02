# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:42:49 2019

@author: kim85
"""

import logging
import sys

FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —"
    "%(funcName)s:%(lineno)d — %(message)s")


def get_console_handler():
    """DESC"""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler
