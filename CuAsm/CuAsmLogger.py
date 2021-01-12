# -*- coding: utf-8 -*-

import logging
import logging.handlers
import sys
import time

class CuAsmLogger(object):
    __LoggerRepos = {}
    __CurrLogger = None

    # Predefined levels:
    # CRITICAL 50 
    # ERROR 40 
    # WARNING 30 
    # INFO 20 
    # DEBUG 10 
    # NOTSET 0 

    # Custom log levels
    ENTRY      = 28   # main entry of a module, between warning and info
    PROCEDURE  = 25   # procedures of some between warning and info
    SUBROUTINE = 10   # some internal subroutines, currently = debug

    @staticmethod
    def getDefaultLoggerFile(name):
        return '%s.log'%name

    @staticmethod
    def initLogger(name='cuasm', log_file='', file_level=logging.DEBUG, file_backup_count=3, stdout_level=logging.WARNING):
        if name in CuAsmLogger.__LoggerRepos:
            CuAsmLogger.__CurrLogger = CuAsmLogger.__LoggerRepos[name]
            print('CuAsmLogger %s already exists! Skipping init...' % name)
            return
        
        logger = logging.getLogger(name)
        CuAsmLogger.__CurrLogger

        fmt = logging.Formatter('%(asctime)s - %(levelname)8s - %(message)s', datefmt='%Y%m%d %H:%M:%S')
        if log_file is not None:
            if len(log_file) == 0:
                log_file = CuAsmLogger.getDefaultLoggerFile(name)

            rfh = logging.handlers.RotatingFileHandler(log_file, backupCount=file_backup_count)
            rfh.setFormatter(fmt)
            rfh.setLevel(file_level)
            logger.addHandler(rfh)

        if stdout_level is not None:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            sh.setLevel(stdout_level)
            logger.addHandler(sh)
    
    @staticmethod
    def setActiveLogger(name):
        if name in CuAsmLogger.__LoggerRepos:
            CuAsmLogger.__CurrLogger = CuAsmLogger.__LoggerRepos[name]
        else:
            print('CuAsmLogger %s does not exist! Keeping current logger...' % name)

    @staticmethod
    def logDebug(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.debug(msg, *args, **kwargs)
        

    @staticmethod
    def logInfo(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.info(msg, *args, **kwargs)
        

    @staticmethod
    def logWarning(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.warning(msg, *args, **kwargs)
        

    @staticmethod
    def logError(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.error(msg, *args, **kwargs)
        
    @staticmethod
    def logCritical(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.critical(msg, *args, **kwargs)


    @staticmethod
    def logEntry(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.log(CuAsmLogger.ENTRY, msg, *args, **kwargs)
        

    @staticmethod
    def logProcedure(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.log(CuAsmLogger.PROCEDURE, msg, *args, **kwargs)
        
    
    @staticmethod
    def logSubroutine(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.log(CuAsmLogger.SUBROUTINE, msg, *args, **kwargs)
        

    @staticmethod
    def log(level, msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.log(level, msg, *args, **kwargs)
        

    @staticmethod
    def logTimeIt(func):
        def wrapper(*args, **kwargs):
            CuAsmLogger.logProcedure('Running %s...'%func.__qualname__)
            t0 = time.time()
            ret = func(*args, **kwargs)
            t1 = time.time()
            CuAsmLogger.logProcedure('Func %s completed! Time=%8.4f secs.'%(func.__qualname__, t1-t0))
            
            return ret

        return wrapper


