import os
import time


def get_root_location() -> str:
    """
    获取项目根目录的路径
    """
    return os.path.dirname(os.path.abspath(__file__))


def get_log_location() -> str:
    """
    获取日志文件的路径
    """
    return os.path.join(get_root_location(), 'logs')


def get_result_location() -> str:
    """
    获取结果文件保存路径
    """
    return os.path.join(get_root_location(), 'results')


def get_log_file_with_timestamp(filename: str):
    """
    获取带时间戳的日志文件名
    """
    return f"{filename}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.txt"


class Logger:
    def __init__(self):
        self.log_path = get_log_location()
        self.log_file_name = get_log_file_with_timestamp('log')

    def printLog(self, message: str):
        """
        用print打印日志并且输出到log中
        """
        print(message)
        with open(os.path.join(self.log_path, self.log_file_name), 'a', encoding='utf-8') as f:
            f.write(f"[time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]:" + message + '\n')


def split_corpus(text: str):
    """
    将长文本切分为多个短文本
    """
    if "." in text:
        split_note = "."
    else:
        split_note = "。"
    return text.split(split_note)


if __name__ == '__main__':
    print(get_root_location())
    print(get_log_location())