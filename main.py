import logging

logging.basicConfig(
    level=logging.INFO,
    datefmt= "%Y-%m-%d %H:%M:%S",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="app.log",encoding="utf-8",mode= "a")
    ],

)

logging.info("测试")