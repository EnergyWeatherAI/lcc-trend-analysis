import logging
from typing import Optional


# Global flag to track if logging has been configured
_logging_configured = False


def setup_logging(level=logging.INFO, force_reconfigure=False):
    """Setup logging configuration.
    
    Args:
        level: Logging level (default: logging.INFO)
        force_reconfigure: If True, reconfigure even if already configured
    """
    global _logging_configured
    
    if _logging_configured and not force_reconfigure:
        return
        
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=force_reconfigure
    )
    logging.captureWarnings(True)
    _logging_configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance, ensuring logging is configured.
    
    Args:
        name: Logger name (default: calling module's name)
    
    Returns:
        Logger instance
    """
    if not _logging_configured:
        setup_logging()
    
    if name is None:
        # Get the caller's module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', __name__)
        else:
            name = __name__
    
    return logging.getLogger(name)