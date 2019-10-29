#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author:      he.zhiming
#

from __future__ import unicode_literals, absolute_import

import logging
import logging.config
import logging.handlers
from datetime import datetime
import os


class _InfoFilter(logging.Filter):
    def filter(self, record):
        """only use INFO

        筛选, 只需要 INFO 级别的log

        :param record:
        :return:
        """
        if logging.INFO <= record.levelno < logging.ERROR:
            # 已经是INFO级别了
            # 然后利用父类, 返回 1
            return super().filter(record)
        else:
            return 0


def _get_filename(*, basename='app.log', log_level='info'):
    date_str = datetime.today().strftime('%Y%m%d')
    pidstr = str(os.getpid())
    return ''.join((
        date_str, '-', pidstr, '-', log_level, '-', basename,))


class _LogFactory:
    # 每个日志文件，使用 2GB
    _SINGLE_FILE_MAX_BYTES = 2 * 1024 * 1024 * 1024
    # 轮转数量是 10 个
    _BACKUP_COUNT = 10

    # 基于 dictConfig，做再次封装
    _LOG_CONFIG_DICT = {
        'version': 1,

        'disable_existing_loggers': False,

        'formatters': {
            # 开发环境下的配置
            'dev': {
                'class': 'logging.Formatter',
                'format': ('%(levelname)s %(asctime)s %(created)f %(name)s %(module)s [%(processName)s %(threadName)s] '
                           '[%(filename)s %(lineno)s %(funcName)s] %(message)s')
            },
            # 生产环境下的格式(越详细越好)
            'prod': {
                'class': 'logging.Formatter',
                'format': ('%(levelname)s %(asctime)s %(created)f %(name)s %(module)s %(process)d %(thread)d '
                           '%(filename)s %(lineno)s %(funcName)s %(message)s')
            }

            # ? 使用UTC时间!!!

        },

        # 针对 LogRecord 的筛选器
        'filters': {
            'info_filter': {
                '()': _InfoFilter,

            }
        },

        # 处理器(被loggers使用)
        'handlers': {
            'console': {  # 按理来说, console只收集ERROR级别的较好
                'class': 'logging.StreamHandler',
                'level': 'ERROR',
                'formatter': 'dev'
            },

            'file': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': _get_filename(log_level='info'),
                'maxBytes': _SINGLE_FILE_MAX_BYTES,  # 2GB
                'encoding': 'UTF-8',
                'backupCount': _BACKUP_COUNT,
                'formatter': 'dev',
                'delay': True,    
                'filters': ['info_filter', ]  # only INFO, no ERROR            
            },
            'file_error': {
                'level': 'ERROR',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': _get_filename(log_level='error'),
                'maxBytes': _SINGLE_FILE_MAX_BYTES,  # 2GB
                'encoding': 'UTF-8',
                'backupCount': _BACKUP_COUNT,
                'formatter': 'dev',
                'delay': True,                
            },

        },

        # 真正的logger(by name), 可以有丰富的配置
        'loggers': {
            'SAMPLE_LOGGER': {
                 # 输送到3个handler，它们的作用分别如下
                 #   1. console：控制台输出，方便我们直接查看，只记录ERROR以上的日志就好
                 #   2. file： 输送到文件，记录INFO以上的日志，方便日后回溯分析
                 #   3. file_error：输送到文件（与上面相同），但是只记录ERROR级别以上的日志，方便研发人员排错
                'handlers': ['console', file', 'file_error'],
                'level': 'INFO'
            },
        },
    }

    logging.config.dictConfig(_LOG_CONFIG_DICT)

    @classmethod
    def get_logger(cls, logger_name):
        return logging.getLogger(logger_name)

# 一个示例
SAMPLE_LOGGER = _LogFactory.get_logger('SAMPLE_LOGGER')
# 示例——debugger，需要先配置好（如同SAMPLE_LOGGER一样）
DEBUGGER = _LogFactory.get_logger('CONSOLE')
# 软件项目一般是分层的，所以可以每一层放置一个logger，各司其职，这里是一个示例
SOME_BASE_LIB_LOGGER = _LogFactory.get_logger('SOME_BASE_LIB_LOGGER')
