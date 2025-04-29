from torch import nn
import torch
import torch.nn.functional as F

class SoftLabelCrossEntropy(nn.Module):
    """支持软标签的交叉熵损失"""
    def __init__(self, soft_label_map=None, num_classes=None, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        self.soft_label_map = soft_label_map
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        """
        inputs: 网络输出logits (B, C, H, W)
        targets: 软标签概率分布 (B, C, H, W)
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        if self.class_weights is not None:
            # 添加类别权重
            weights = self.class_weights.to(inputs.device)
            log_probs = log_probs * weights.view(1, -1, 1, 1)
            
        loss = -(targets * log_probs).sum(dim=1)
        return loss.mean()



    def _convert_to_soft_labels(self, hard_labels):
        """动态硬标签->软标签转换核心方法"""
        batch_size, H, W = hard_labels.shape
        device = hard_labels.device
        
        # 初始化全零概率分布
        soft_labels = torch.zeros(
            batch_size, self.num_classes, H, W,
            device=device, dtype=torch.float32
        )
        
        # 处理每个需要软化的类别
        for src_class in self.soft_label_map:
            # 创建当前类别的二进制掩码
            mask = (hard_labels == src_class).float()  # [B, H, W]
            
            # 分配概率到目标类别
            for dst_class, prob in self.soft_label_map[src_class]:
                soft_labels[:, dst_class] += mask * prob # mask * prob 当前源类别中分配给目标类别的概率
                
        # 处理非混淆类别（保持one-hot）
        for class_idx in range(self.num_classes):
            if class_idx not in self.soft_label_map:
                mask = (hard_labels == class_idx).float()
                soft_labels[:, class_idx] += mask
                
        return soft_labels


if __name__ == '__main__':
    # 配置
    soft_label_map = {
        0: [[1, 0.8], [2, 0.2]],
        1: [[0, 0.5], [1, 0.3], [2, 0.2]]
    }
    loss = SoftLabelCrossEntropy(soft_label_map, 3)
    hard_labels = torch.tensor([
        [[1, 2],
        [0, 1]],
        [[0, 0],
        [2, 1]],
    ])

    soft_labels = loss._convert_to_soft_labels(hard_labels)
    print(soft_labels)