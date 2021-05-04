# -*- coding: utf-8 -*-

import logging
import logging.handlers
import sys
import time
import os


class CuAsmLogger(object):
    ''' A logger private to current module.

        A customized logging style is used to show the progress better, 
        without affecting the logging of other modules.

    '''
    __LoggerRepos = {}
    __CurrLogger = None
    __IndentLevel = 0
    __IndentString = ''

    # Predefined levels:
    # CRITICAL 50 
    # ERROR    40 
    # WARNING  30 
    # INFO     20 
    # DEBUG    10 
    # NOTSET    0 

    # Custom log levels

    ENTRY      = 35   # main entry of a module
    PROCEDURE  = 25   # procedures of some module
    SUBROUTINE = 15   # some internal subroutines

    @staticmethod
    def getDefaultLoggerFile(name):
        return '%s.log'%name

    @staticmethod
    def initLogger(name='cuasm', log_file=None, file_level=logging.DEBUG, file_backup_count=3, stdout_level=25):
        if name in CuAsmLogger.__LoggerRepos:
            CuAsmLogger.__CurrLogger = CuAsmLogger.__LoggerRepos[name]
            print('CuAsmLogger %s already exists! Skipping init...' % name)
            return
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        fmt = logging.Formatter('%(asctime)s - %(message)s')
        if log_file is not None:
            if len(log_file) == 0:
                log_file = CuAsmLogger.getDefaultLoggerFile(name)

            rfh = logging.handlers.RotatingFileHandler(log_file, mode='a', backupCount=file_backup_count)

            # default mode is 'a', but we may want a new log for every run, but still keeping old logs as backup.
            if os.path.exists(log_file):
                rfh.doRollover()

            rfh.setFormatter(fmt)
            rfh.setLevel(file_level)
            
            logger.addHandler(rfh)

        if stdout_level is not None:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            sh.setLevel(stdout_level)
            logger.addHandler(sh)

        # 
        CuAsmLogger.__LoggerRepos[name] = logger
        CuAsmLogger.__CurrLogger = logger
    
    @staticmethod
    def setActiveLogger(name):
        if name in CuAsmLogger.__LoggerRepos:
            CuAsmLogger.__CurrLogger = CuAsmLogger.__LoggerRepos[name]
        else:
            print('CuAsmLogger %s does not exist! Keeping current logger...' % name)

    @staticmethod
    def logDebug(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.debug('   DEBUG - ' + msg, *args, **kwargs)
        
    @staticmethod
    def logInfo(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.info('    INFO - ' + msg, *args, **kwargs)
        
    @staticmethod
    def logWarning(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.warning(' WARNING - ' + msg, *args, **kwargs)
        
    @staticmethod
    def logError(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.error('   ERROR - ' + msg, *args, **kwargs)
        
    @staticmethod
    def logCritical(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.critical('CRITICAL - ' + msg, *args, **kwargs)

    @staticmethod
    def logEntry(msg, *args, **kwargs):
        full_msg = '   ENTRY - ' + CuAsmLogger.__IndentString + msg
        CuAsmLogger.__CurrLogger.log(CuAsmLogger.ENTRY, full_msg, *args, **kwargs)
        

    @staticmethod
    def logProcedure(msg, *args, **kwargs):

        full_msg = '    PROC - ' + CuAsmLogger.__IndentString + msg
        CuAsmLogger.__CurrLogger.log(CuAsmLogger.PROCEDURE, full_msg, *args, **kwargs)
        
    
    @staticmethod
    def logSubroutine(msg, *args, **kwargs):
        full_msg = '     SUB - ' + CuAsmLogger.__IndentString + msg
        CuAsmLogger.__CurrLogger.log(CuAsmLogger.SUBROUTINE, full_msg, *args, **kwargs)
        

    @staticmethod
    def logLiteral(msg, *args, **kwargs):
        full_msg = '         - ' + CuAsmLogger.__IndentString + msg
        CuAsmLogger.__CurrLogger.log(CuAsmLogger.PROCEDURE, full_msg, *args, **kwargs)
        

    @staticmethod
    def log(level, msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.log(level, msg, *args, **kwargs)
        

    @staticmethod
    def logTimeIt(func):
        ''' Logging of a (usually) long running function.

        '''
        def wrapper(*args, **kwargs):
            CuAsmLogger.logLiteral('Running %s...'%func.__qualname__)
            CuAsmLogger.incIndent()
            
            t0 = time.time()
            ret = func(*args, **kwargs)
            t1 = time.time()

            CuAsmLogger.decIndent()
            CuAsmLogger.logLiteral('Func %s completed! Time=%8.4f secs.'%(func.__qualname__, t1-t0))
            
            return ret

        return wrapper
    
    @staticmethod
    def logIndentIt(func):
        '''
        '''
        def wrapper(*args, **kwargs):
            CuAsmLogger.incIndent()
            ret = func(*args, **kwargs)
            CuAsmLogger.decIndent()
            
            return ret

        return wrapper

    @staticmethod
    def logTraceIt(func):
        '''
        '''
        def wrapper(*args, **kwargs):
            CuAsmLogger.logLiteral('Running %s...'%func.__qualname__)
            CuAsmLogger.incIndent()
            
            ret = func(*args, **kwargs)
            CuAsmLogger.decIndent()

            return ret

        return wrapper

    @staticmethod
    def incIndent():
        CuAsmLogger.__IndentLevel += 1
        CuAsmLogger.__IndentString = '    ' * CuAsmLogger.__IndentLevel

    @staticmethod
    def decIndent():
        CuAsmLogger.__IndentLevel -= 1
        if CuAsmLogger.__IndentLevel < 0:
            CuAsmLogger.__IndentLevel = 0
        CuAsmLogger.__IndentString = '    ' * CuAsmLogger.__IndentLevel
    
    @staticmethod
    def resetIndent(val=0):
        if val<0:
            val = 0
        CuAsmLogger.__IndentLevel = val
        CuAsmLogger.__IndentString = '    ' * CuAsmLogger.__IndentLevel

    @staticmethod
    def setLevel(level):
        CuAsmLogger.__CurrLogger.setLevel(level)
    
    @staticmethod
    def disable():
        CuAsmLogger.__CurrLogger.setLevel(logging.ERROR)

# Init a default logger when the module is imported
CuAsmLogger.initLogger()
