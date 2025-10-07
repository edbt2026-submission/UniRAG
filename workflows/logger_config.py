import logging

def setup_logger(log_file_path):
    """
    设置日志记录器，将日志同时输出到控制台和文件，并使用 UTF-8 编码保存日志文件。

    :param log_file_path: 日志文件的路径
    :return: 配置好的日志记录器
    """
    # 创建一个日志记录器
    logger = logging.getLogger("UniRAG")
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG

    # 创建一个文件处理器，用于将日志写入文件，并指定 UTF-8 编码
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 文件日志级别为DEBUG

    # 创建一个控制台处理器，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台日志级别为INFO

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger