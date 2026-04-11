import logging
from functools import wraps


def get_logger(name: str, filepath: str = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 控制台
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    # 文件
    if filepath:
        file_handler = logging.FileHandler(filepath)
        logger.addHandler(file_handler)

    return logger


def log(enter_message: str = "", exit_message: str = ""):
    return _Log(enter_message=enter_message, exit_message=exit_message)


class _Log:
    def __init__(
        self,
        enter_message: str = "",
        exit_message: str = ""
    ):
        self.enter_message = enter_message
        self.exit_message = exit_message

    def __enter__(self):
        if self.enter_message:
            print(self.enter_message)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.exit_message:
            print(self.exit_message)
        print("")
        return False

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with _Log(self.enter_message, self.exit_message):
                return func(*args, **kwargs)
            return None

        return wrapper
