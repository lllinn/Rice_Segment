import logging
from typing import Dict
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
import os

class CustomLogger(TensorBoardLogger):
    """自定义日志记录器，集成TensorBoard和文件日志"""
    
    def __init__(self, config, **kwargs):
        """
        初始化类实例。
        
        Args:
            config (dict): 包含日志目录和实验名称的配置字典。
            **kwargs: 其他关键字参数，将传递给父类的构造函数。
        
        Returns:
            None
        
        """
        super().__init__(
            save_dir=config['log_dir'],
            name=config['experiment_name'],
            **kwargs
        )
        self._setup_file_logger(config)
        
    def _setup_file_logger(self, config):
            """
            配置并初始化文件日志记录器。
            
            Args:
                config (dict): 包含配置信息的字典，其中应包含 'log_dir' 和 'experiment_name' 两个键。
            
            Returns:
                None
            
            """
            self.file_logger = logging.getLogger(__name__)
            self.file_logger.setLevel(logging.INFO)
            
            log_dir = config['log_dir']
            os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在
            
            handler = logging.FileHandler(
                f"{log_dir}/{config['experiment_name']}.log"
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.file_logger.addHandler(handler)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        记录指标数据到日志中。
        
        Args:
            metrics (Dict[str, float]): 包含所有指标名称和对应值的字典。
            step (int, optional): 当前步骤，默认为None。
        
        Returns:
            None
        
        """
        super().log_metrics(metrics, step)
        for k, v in metrics.items():
            self.file_logger.info(f"{k}: {v}")