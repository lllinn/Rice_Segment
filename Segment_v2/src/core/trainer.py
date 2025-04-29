from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger

class SegmentationTrainer:
    """封装训练流程的完整训练器"""
    
    def __init__(self, config):
        self.config = config
        self._init_callbacks()
        self._init_logger()
        
    def _init_callbacks(self):
        # 初始化 ModelCheckpoint 回调函数，用于保存最优模型
        self.checkpoint_callback = ModelCheckpoint(
            # 监视的变量，此处为验证集的损失值
            # monitor='val/acc',   
            monitor='val/IoU',   
            # 保存模型的文件名
            filename=f'best_model'+"-{epoch}-{val/IoU}-{val/acc}",
            # 保存最优的模型数量
            save_top_k=1,
            # 保存策略，此处为取最小的损失值为TopK
            mode='max',
            verbose=True, # 是否打印日志
        )
    
        # 初始化 EarlyStopping 回调函数，用于在验证集性能不再提升时停止训练
        self.early_stop_callback = EarlyStopping(
            # 监视的变量，此处为验证集的损失值
            # monitor='val/acc',
            monitor='val/IoU',
            # 容忍多少个epoch性能不再提升
            patience=self.config.get('early_stop_patience', 10),
            # 保存策略，此处为取最小的损失值
            mode='max',
            verbose=True, # 是否打印日志
        )

        # 初始化 LearningRateMonitor 回调函数，用于记录每个epoch的学习率
        self.lr_monitor = LearningRateMonitor(logging_interval='epoch')

    def _init_logger(self):
        self.logger = TensorBoardLogger(
            save_dir=self.config['log_dir'],
            name=self.config['experiment_name']
        )
        
    def fit(self, model, data_module):
        """
        使用提供的模型和数据模块进行训练。
    
        Args:
            model (pl.LightningModule): 要训练的模型。
            data_module (pl.LightningDataModule): 数据模块，包含训练和验证数据。
    
        Returns:
            None
    
        """
        # 初始化训练器
        trainer = Trainer(
            # 设置最大训练轮数
            max_epochs=self.config['epochs'],
            # 设置使用的GPU数量
            gpus=self.config.get('gpus', -1),
            # 设置日志记录器
            logger=self.logger,
            # 设置回调函数列表
            callbacks=[
                self.checkpoint_callback,
                self.early_stop_callback,
                self.lr_monitor
            ],
            # 设置训练精度
            precision=self.config['precision'], # 训练精度
            # 设置梯度累积的批次数
            accumulate_grad_batches=self.config['accumulate_steps'],
            # 设置训练过程是否确定
            deterministic=False,  # 这里用True会报错
            # 选取train dataset一定比例的数据进行训练
            limit_train_batches=self.config['train_batch_ratio'],
            limit_val_batches=self.config['val_batch_ratio'],
            limit_test_batches=self.config['test_batch_ratio'],
        )
        # 使用训练器进行训练
        trainer.fit(model, *data_module)