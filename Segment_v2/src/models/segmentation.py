import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from ..utils.metrics import SegmentationMetrics
import pandas as pd
import os
from ..losses import soft_label_ce
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import csv
from torchinfo import summary
from thop import profile
from ptflops import get_model_complexity_info
import time

class SegmentationModel(pl.LightningModule):
    """支持多种分割架构的PyTorch Lightning模块"""
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters("config")

        # 获取ignore_index
        self.ignore_index = config.get("ignore_index", None)
        
        # 初始化模型
        self.model = smp.create_model(
            config['arch'],
            encoder_name=config['encoder'],
            in_channels=config['in_channels'],
            classes=config['num_classes'],
            encoder_weights=config['pretrained']
        )

        # 初始化指标
        self.train_metrics = SegmentationMetrics(config['num_classes'], ignore_index=self.ignore_index)
        self.val_metrics = SegmentationMetrics(config['num_classes'], ignore_index=self.ignore_index)
        self.test_metrics = SegmentationMetrics(config['num_classes'], ignore_index=self.ignore_index)
        
        # 配置损失函数
        # self.loss_fn = smp.losses.DiceLoss(
        #     smp.losses.MULTICLASS_MODE, 
        #     from_logits=True
        # )

        # 配置损失函数
        if config['loss_type'] == "focal_loss" and not config['soft_label']:
            self.loss_fn = smp.losses.FocalLoss(
                mode="multiclass", ignore_index=self.ignore_index
            )
        elif config['loss_type'] == "crossentropy_loss":
            # 不进行标签平滑
            self.loss_fn = smp.losses.SoftCrossEntropyLoss(smooth_factor=0, ignore_index=self.ignore_index)
        else:
            self.loss_fn = smp.losses.SoftCrossEntropyLoss(smooth_factor=0, ignore_index=self.ignore_index)

            
        # 软标签损失函数
        # self.soft_label_loss = soft_label_ce.SoftLabelCrossEntropy()



        # self.loss_fn = smp.losses.FocalLoss(
        #     smp.losses.MULTICLASS_MODE, 
        # )

    def forward(self, x):
        # 归一化, 在trainsform中已经归一化了
        # image = (image - self.mean) / self.std
        return self.model(x)

    def _shared_step(self, batch, stage):
        images, masks = batch
        logits = self.forward(images)
        loss = self.loss_fn(logits, masks)
        
        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)

        # 更新指标
        metrics = getattr(self, f"{stage}_metrics")
        metrics.add_batch(preds, masks)

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        if not self.hparams.config['soft_label']: # 没有启动软标签
            # raise Exception('Not implemented yet')
            loss = self._shared_step(batch, "train")
            return loss

        # 软标签实现
        images, hard_labels = batch

        # 转换软标签
        soft_labels = self._convert_to_soft_labels(hard_labels)
        
        # 前向传播
        logits = self.forward(images)
        # 计算损失
        loss = self.soft_label_loss(logits, soft_labels)

        # 使用硬标签更新指标
        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)

        # 更新指标
        self.train_metrics.add_batch(preds, hard_labels)

        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "test")
        return loss

    def _shared_epoch_end(self, outputs, stage):
        # 计算指标
        metrics = {}

        # 记录loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics[f'{stage}/loss'] = avg_loss
        
        # 计算accuracy
        stage_metrics = getattr(self, f"{stage}_metrics")        
        metrics[f'{stage}/acc'] = stage_metrics.compute_overall_accuracy()

        # 计算IoU指标, 取最后一个类别的IoU, rice_mild IoU
        metrics[f'{stage}/IoU'] = stage_metrics.compute_iou().numpy()[self.hparams.config['IoU_index']].mean()

        # 计算mIoU指标
        metrics[f'{stage}/mIoU'] = stage_metrics.compute_mean_iou()

        # 手动指定步长为当前 epoch，并记录到 TensorBoard
        current_epoch = self.trainer.current_epoch    
        for key, value in metrics.items():        
            self.logger.experiment.add_scalar(f"manual/{key}", value, current_epoch)

        self.log_dict(metrics, on_epoch=True, on_step=False) # 以epoch为单位记录

        # 保存到csv表格的核心逻辑
        self._save_metrics_to_csv(stage, current_epoch, metrics)

        # 重置指标
        stage_metrics.reset()

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        # self._shared_epoch_end(outputs, "test")
        metrics = {}
        metrics['mIoU'] = self.test_metrics.compute_mean_iou().numpy()
        metrics['OA'] = self.test_metrics.compute_overall_accuracy().numpy()
        metrics['Precision'] = self.test_metrics.compute_precision().numpy()
        metrics['Recall'] = self.test_metrics.compute_recall().numpy()
        metrics['F1'] = self.test_metrics.compute_f1score().numpy()
        metrics['IoU'] = self.test_metrics.compute_iou().numpy()

        # 保存混淆矩阵
        self.test_metrics.plot_confusion_matrix(
            self.hparams.config["class_names"],
            os.path.join(self.logger.log_dir, "confusion_matrix.png")
        )

        self.test_metrics.plot_confusion_matrix(
            self.hparams.config["class_names"],
            os.path.join(self.logger.log_dir, "opposite_confusion_matrix.png"),
            normalize='pred'
        )

        self.test_metrics.plot_confusion_matrix(
            self.hparams.config["class_names"],
            os.path.join(self.logger.log_dir, "count"),
            normalize=None,
        )
        
        # 保存到文件中
        self.save_report(metrics, output_path=self.logger.log_dir)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=float(self.hparams.config['lr']),
            weight_decay=float(self.hparams.config.get('weight_decay', 1e-4))
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.hparams.config['T_Max'], # 每个Epoch更新一次学习率
            eta_min=float(self.hparams.config['eta_min'])
            # T_max=20 # 每个Epoch更新一次学习率
        )
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.01,  # 初始学习率为 lr * 0.01
            total_iters=5       # 5个epoch后达到完整lr
        )
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[5]  # 第5个epoch后切换为CosineAnnealingLR
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # 更新周期为epoch
                "frequency": 1,        # 每个周期更新一次
            }
        }
    
    def save_report(self, metrics, output_path: str):
        """保存评估报告"""
        classwise_data = []
        # 纵轴是指标，横轴是类别
        vertical_name = ["Precision", "Recall", "F1", "IoU"]
        for name in vertical_name:
            class_metircs = metrics[name]
            classwise_data.append(
                [name] + class_metircs.tolist()
            )
        
        # 横轴名称
        horizontal_names = self.hparams.config["class_names"].copy() # 防止修改原始列表
        horizontal_names.insert(0, "Index")
        # 创建DataFrame
        df_class = pd.DataFrame(
            classwise_data,
            columns=horizontal_names
        )

        df_class = df_class.round(4) # 保留4位小数

        df_overall = pd.DataFrame({
            "OA": [metrics["OA"].round(4)],
            "mIoU": [metrics["mIoU"].round(4)]
        })
        
        # df_overall = df_overall.round(4)
        
        # 保存到csv文件
        df_class.to_csv(os.path.join(output_path, "class.csv"), index=False)
        df_overall.to_csv(os.path.join(output_path, "overall.csv"), index=False)
    
    def _convert_to_soft_labels(self, hard_labels):
        """动态硬标签->软标签转换核心方法"""
        batch_size, H, W = hard_labels.shape
        device = hard_labels.device
        
        # 初始化全零概率分布
        soft_labels = torch.zeros(
            batch_size, self.hparams.config['num_classes'], H, W,
            device=device, dtype=torch.float32
        )
        
        # 处理每个需要软化的类别
        for src_class in self.hparams.config['soft_label_map']:
            # 创建当前类别的二进制掩码
            mask = (hard_labels == src_class).float()  # [B, H, W]
            
            # 分配概率到目标类别
            for dst_class, prob in self.hparams.config['soft_label_map'][src_class]:
                soft_labels[:, dst_class] += mask * prob # mask * prob 当前源类别中分配给目标类别的概率
                
        # 处理非混淆类别（保持one-hot）
        for class_idx in range(self.hparams.config['num_classes']):
            if class_idx not in self.hparams.config['soft_label_map']:
                mask = (hard_labels == class_idx).float()
                soft_labels[:, class_idx] += mask
                
        return soft_labels
    
    def _save_metrics_to_csv(self, stage: str, epoch: int, metrics):
        """将指标保存到CSV文件，每个阶段独立文件"""
        # 生成文件名（如 'train_metrics.csv'）
        filename = f"{stage}_metrics.csv"
        stage_metrics = getattr(self, f"{stage}_metrics")        

        # 计算每个的IoU
        metrics[f'{stage}/Class_IoU'] = stage_metrics.compute_iou().numpy()    
        # 获取日志目录路径（兼容不同Logger类型）
        log_dir = (
            self.logger.log_dir if hasattr(self.logger, 'log_dir') 
            else self.trainer.default_root_dir
        )
        csv_path = os.path.join(log_dir, filename)
        
        # 定义CSV列顺序及数据转换
        class_names = self.hparams.config['class_names']
        columns = ['epoch', 'loss', 'acc', 'mIoU','iou']
        columns.extend(class_names)
        row_data = {
            'epoch': epoch,
            'loss': metrics[f'{stage}/loss'].item(),      # Tensor转float
            'acc': metrics[f'{stage}/acc'].item(),        # 假设acc是Tensor
            'mIoU': metrics[f'{stage}/mIoU'].item(),      # 
            'iou': float(metrics[f'{stage}/IoU'])         # 兼容numpy类型
            }
        for i, class_name in enumerate(class_names):
            row_data[class_name] = metrics[f'{stage}/Class_IoU'][i]
        
        # 检查文件是否存在，决定是否写入表头
        file_exists = os.path.exists(csv_path)
        
        # 写入文件（追加模式）
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            
            # 新文件需要写入表头
            if not file_exists:
                writer.writeheader()
                
            # 写入当前epoch数据
            writer.writerow(row_data)

    def calcute_model_performance(self):
    # 计算模型参数量, FLOPS, FPS
        # 1. 计算参数量
        input_size = (1, 3, 640, 640)
        stats = summary(self.model, input_size, verbose=0)
        self.params_m = round(stats.total_params / 1e6, 2)
        
        # 2. 计算FLOPS
        input_sample = torch.randn(*input_size).to(self.device)
        self.model = self.model.to(self.device)
        flops, _ = profile(self.model, inputs=(input_sample,), verbose=False)
        self.flops = round(flops / 1e9, 2)  # 转换为GigaFLOPS
        
        # 3. 测量FPS
        self.fps = self._measure_fps()
        

        input_shape = (3, 640, 640)
        macs, params = get_model_complexity_info(self.model, input_shape, as_strings=True, print_per_layer_stat=True)

        print("Params:", self.params_m, "M")
        print("FLOPs:", self.flops, "GFLOP/s")
        print("FPS:", self.fps, "images/sec")

        print(f"计算复杂度: {macs}")
        print(f"参数数量: {params}") 


    
    def _measure_fps(self, batch_size=32, warmup=100, repeats=100):
        self.model.eval()
        
        input_tensor = torch.randn(batch_size, 3, 640, 640).cuda()
        self.model = self.model.cuda()

        # 预热
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(input_tensor)
        
        # 计时
        start = time.time()
        for _ in range(repeats):
            with torch.no_grad():
                _ = self.model(input_tensor)
        total_time = time.time() - start
        
        return round(repeats * batch_size / total_time, 1)

