import logging

logging.basicConfig(
    filename="skeleton.log",
    encoding="utf-8",
    filemode="w",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S:%MS",
    level=logging.INFO
)