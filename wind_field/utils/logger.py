# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 20:44:52 2026

@author: 38036
"""

import logging
import sys

class Logger:
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        if name in cls._loggers:
            return cls._loggers[name]
            
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)-8s | %(levelname)-7s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        cls._loggers[name] = logger
        return logger