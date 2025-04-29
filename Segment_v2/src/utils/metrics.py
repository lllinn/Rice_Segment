import torch
import numpy as np
import matplotlib.pyplot as plt
import os


class SegmentationMetrics:
    """多类语义分割评估指标计算"""
    
    def __init__(self, num_classes, device='cuda', ignore_index=None):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
        self.ignore_index = ignore_index

    def update(self, pred, target):
        # mask = (target >= 0) & (target < self.num_classes)
        # labels = self.num_classes * target[mask] + pred[mask]
        # counts = torch.bincount(labels, minlength=self.num_classes**2)
        # self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

        # Ensure tensors are on the correct device and flatten
        pred = pred.flatten()
        target = target.flatten()

        # --- Modification Start ---
        # Create a mask for valid pixels (not ignore_index and within class bounds)
        valid_mask = (target != self.ignore_index) if self.ignore_index is not None else torch.ones_like(target, dtype=torch.bool)
        valid_mask &= (target >= 0) & (target < self.num_classes) # Also ensure target is within bounds

        # Apply the mask
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        # --- Modification End ---


        # Calculate confusion matrix entries only for valid pixels
        # Note: Need target_valid to be long for bincount index
        labels = self.num_classes * target_valid.long() + pred_valid.long()
        counts = torch.bincount(labels, minlength=self.num_classes**2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)



    def add_batch(self, preds, targets):
        # for batch_idx in range(len(preds)):
        #     self.update(preds[batch_idx], targets[batch_idx])
        self.update(preds, targets)

    def _compute_safe_division(self, numerator, denominator):
        # Helper for safe division, returns tensor on cpu
        # Ensure denominator is float for division
        denominator = denominator.float()
        # Set zero denominators to a small epsilon to avoid NaN
        denominator[denominator == 0] = 1e-8
        result = numerator.float() / denominator
        return result.cpu()


    def compute_precision(self):
        # return (torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(0) + 1e-8)).cpu()
        # Precision = TP / (TP + FP) = diag / col_sum
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(0) - tp # Sum of column - diagonal
        return self._compute_safe_division(tp, tp + fp)


    def compute_recall(self):
        # return (torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(1) + 1e-8)).cpu()
        tp = torch.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(1) - tp # Sum of row - diagonal
        return self._compute_safe_division(tp, tp + fn)
    

    def compute_f1score(self):
        # precision = self.compute_precision()
        # recall = self.compute_recall()
        # f1 = 2 * precision * recall / (precision + recall + 1e-8)
        # return f1
        precision = self.compute_precision() # Move back to device for calc
        recall = self.compute_recall()
        # Ensure precision + recall is float for division
        denominator = (precision + recall).float()
        denominator[denominator == 0] = 1e-8 # Avoid NaN
        f1 = 2 * precision * recall / denominator
        return f1.cpu() # Return on CPU


    def compute_iou(self):
        # intersection = torch.diag(self.confusion_matrix)
        # union = self.confusion_matrix.sum(0) + self.confusion_matrix.sum(1) - intersection
        # return (intersection / (union + 1e-8)).cpu()
        # IoU = TP / (TP + FP + FN) = intersection / union
        intersection = torch.diag(self.confusion_matrix) # TP
        union = self.confusion_matrix.sum(0) + self.confusion_matrix.sum(1) - intersection # TP+FP + TP+FN - TP = TP+FP+FN
        return self._compute_safe_division(intersection, union)


    def compute_overall_accuracy(self):
        # correct = torch.diag(self.confusion_matrix).sum()
        # total = self.confusion_matrix.sum()
        # return (correct / total).cpu()
        # OA = Correct / Total = sum(diag) / sum(all)
        correct = torch.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        # Handle case where total might be 0 if all pixels were ignored
        # if total == 0:
        #     return torch.tensor(float('nan')) # Or 0.0, depending on desired behavior
        return (correct.float() / total.float()).cpu()


    def compute_mean_iou(self, ignore_background=True):  # Option to ignore background class (index 0) in mIoU
        # iou = self.compute_iou()
        # return iou.mean().cpu()
        iou = self.compute_iou()
        # Remove NaN values which can occur if a class had zero union
        iou = iou[~torch.isnan(iou)]
        if ignore_background and self.num_classes > 1 and len(iou) > 0:
             # Check if index 0 was included (it shouldn't be if ignore_index=0 was effective)
             # But as a safeguard, or if ignore_index wasn't 0, we might skip index 0 for mIoU.
             # This logic might need refinement based on whether index 0 is truly absent due to ignore_index.
             # If ignore_index=0 was used, the IoU for class 0 should be NaN or based on zero counts.
             # A safer way might be to compute IoU only for relevant classes:
             valid_iou = []
             for i in range(self.num_classes):
                 if i != self.ignore_index: # Explicitly skip ignore_index
                    tp = self.confusion_matrix[i, i]
                    union = self.confusion_matrix[i, :].sum() + self.confusion_matrix[:, i].sum() - tp
                    if union > 0:
                        valid_iou.append(tp.float() / union.float())
             if not valid_iou:
                 return torch.tensor(float('nan'))
             return torch.stack(valid_iou).mean().cpu()
        elif len(iou) == 0:
             return torch.tensor(float('nan')) # Or 0.0
        else:
             return iou.mean().cpu()


    def reset(self):
        self.confusion_matrix.zero_()


    def plot_confusion_matrix(self, class_names, save_path=None, normalize='true'):
        """Plots confusion matrix. normalize can be 'true', 'pred', or None."""
        if self.confusion_matrix.sum() == 0:
             print("Warning: Confusion matrix is empty, skipping plot.")
             return

        cm = self.confusion_matrix.cpu().numpy()
        title = "Confusion Matrix"

        if normalize == 'true':
            # Normalize by true label (row-wise)
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            # Avoid division by zero for rows with no samples
            row_sums[row_sums == 0] = 1
            cm_norm = cm.astype('float') / row_sums
            title = "Normalized Confusion Matrix"
        elif normalize == 'pred':
            # Normalize by predicted label (column-wise)
            col_sums = cm.sum(axis=0)
            # Avoid division by zero for columns with no predictions
            col_sums[col_sums == 0] = 1
            cm_norm = cm.astype('float') / col_sums
            title = "Normalized Confusion Matrix (by Predicted Label)"
        else:
             cm_norm = cm # No normalization
             title = "Confusion Matrix (Counts)"


        # Adjust class_names if ignore_index was used?
        # Decide if you want to show the ignored class in the plot (it should have all zeros)
        # Or filter it out. Let's assume we show all classes specified.
        if len(class_names) != self.num_classes:
             print(f"Warning: Number of class names ({len(class_names)}) does not match num_classes ({self.num_classes}). Using indices.")
             plot_class_names = [str(i) for i in range(self.num_classes)]
        else:
             plot_class_names = class_names

        # plt.figure(figsize=(max(6, self.num_classes * 0.8), max(6, self.num_classes * 0.8))) # Adjust size
        plt.figure(figsize=(8, 8))
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(plot_class_names))
        plt.xticks(tick_marks, plot_class_names, rotation=45, ha='center')
        plt.yticks(tick_marks, plot_class_names)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.grid(False) # Turn off grid lines

        # Print values in cells
        fmt = '.2f' if normalize else 'd' # Format for float or int
        thresh = cm_norm.max() / 2. if normalize else cm.max() / 2. # Threshold for text color

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                # Only display text if value is meaningful (e.g., > 0 for normalized)
                value = cm_norm[i, j]
                if (normalize and value > 1e-4) or (not normalize and value > 0):
                    plt.text(j, i, format(value * 100 if normalize else value, fmt) + ('%' if normalize else ''),
                             ha="center", va="center",
                             color="white" if cm_norm[i, j] > thresh else "black")

        plt.tight_layout() # Adjust layout

        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        # plt.show() # Optionally show plot immediately
        plt.close()
    # def plot_confusion_matrix(self, class_names, save_path=None):
    #     cm = self.confusion_matrix.cpu().numpy()
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    #     plt.title("Normalized Confusion Matrix")
    #     plt.colorbar()
        
    #     tick_marks = np.arange(len(class_names))
    #     plt.xticks(tick_marks, class_names, rotation=45)
    #     plt.yticks(tick_marks, class_names)
        
    #     plt.ylabel('True label')
    #     plt.xlabel('Predicted label')

    #     # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字，保留一位小数
    #     for i in range(np.shape(cm)[0]):
    #         for j in range(np.shape(cm)[1]):
    #             if not np.isnan(cm[i][j]) and cm[i][j] * 100 > 0:
    #                 plt.text(j, i, format(cm[i][j] * 100, '.2f') + '%',
    #                         ha="center", va="center",
    #                         color="white" if cm[i][j] > 0.8 else "black")  # 如果要更改颜色风格，需要同时更改此行
       
    #     if save_path:
    #         if not os.path.exists(os.path.dirname(save_path)):
    #             os.makedirs(os.path.dirname(save_path))
    #         plt.savefig(save_path, bbox_inches='tight')
    #     plt.close()


    # def plot_confusion_matrix_opposite(self, class_names, save_path=None):
    #     cm = self.confusion_matrix.cpu().numpy()
    #     cm = cm.astype('float') / cm.sum(axis=0)
        
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    #     plt.title("Opposite Normalized Confusion Matrix")
    #     plt.colorbar()
        
    #     tick_marks = np.arange(len(class_names))
    #     plt.xticks(tick_marks, class_names, rotation=45)
    #     plt.yticks(tick_marks, class_names)
        
    #     plt.ylabel('True label')
    #     plt.xlabel('Predicted label')

    #     # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字，保留一位小数
    #     for i in range(np.shape(cm)[0]):
    #         for j in range(np.shape(cm)[1]):
    #             if not np.isnan(cm[i][j]) and cm[i][j] * 100 > 0:
    #                 plt.text(j, i, format(cm[i][j] * 100, '.2f') + '%',
    #                         ha="center", va="center",
    #                         color="white" if cm[i][j] > 0.8 else "black")  # 如果要更改颜色风格，需要同时更改此行
       
    #     if save_path:
    #         if not os.path.exists(os.path.dirname(save_path)):
    #             os.makedirs(os.path.dirname(save_path))
    #         plt.savefig(save_path, bbox_inches='tight')
    #     plt.close()



if __name__ == '__main__':
    # Example Usage with ignore_index=0
    num_classes_example = 3
    ignore_index_example = 0 # The class to ignore

    # Example prediction and label tensors including the ignore_index
    # pred: [0, 0, 1, 1, 2, 2, 0, 1]
    # true: [0, 0, 0, 1, 2, 2, 1, 0] -> Ignore index 0. Valid pairs are (1,0), (1,1), (2,2), (2,2), (1,1) ??? Check example logic.
    # Let's use a clearer example:
    # pred: [0, 1, 2, 1, 0, 2]
    # true: [0, 0, 2, 1, 0, 1]  (Class 0 is background/ignored)

    # After filtering out true == 0:
    # pred_valid: [2, 1, 2]
    # true_valid: [2, 1, 1]

    # Confusion matrix (only for classes 1, 2):
    #   Pred: 1  2
    # True
    #  1        1  1   (True=1, Pred=1), (True=1, Pred=2)
    #  2        0  1   (True=2, Pred=2)

    # Expected CM (size 3x3, but only indices 1,2 used):
    # [[0, 0, 0],
    #  [0, 1, 1],
    #  [0, 0, 1]]

    imgPredict = torch.tensor([0, 1, 2, 1, 0, 2]).to("cuda")
    imgLabel = torch.tensor(  [0, 0, 2, 1, 0, 1]).to("cuda") # Note: Changed target for class 1 prediction

    # metric = SegmentationMetrics(num_classes_example, ignore_index=None) # Original
    metric = SegmentationMetrics(num_classes_example, ignore_index=ignore_index_example) # With ignore

    metric.update(imgPredict, imgLabel)

    print("Confusion Matrix (Raw Counts):\n", metric.confusion_matrix.cpu().numpy())
    # Expected Output (based on manual calc above):
    # [[0 0 0]
    #  [0 1 1]
    #  [0 0 1]]

    # Provide actual class names excluding the ignored one? Or include it?
    # Let's provide names for all potential classes (0, 1, 2)
    class_names_example = ['background', 'class1', 'class2']

    # Check if metrics handle the ignored class correctly
    print("Precision:", metric.compute_precision().numpy()) # Should be [nan, 1.0, 0.5] -> TP / (TP+FP) -> [0/0, 1/1, 1/(1+1)]
    print("Recall:", metric.compute_recall().numpy())    # Should be [nan, 0.5, 1.0] -> TP / (TP+FN) -> [0/0, 1/(1+1), 1/1]
    print("F1 Score:", metric.compute_f1score().numpy())  # Should be [nan, 0.666, 0.666]
    print("IoU:", metric.compute_iou().numpy())          # Should be [nan, 0.5, 0.5] -> TP / (TP+FP+FN) -> [0/0, 1/(1+0+1), 1/(1+1+0)]
    print("Overall Accuracy:", metric.compute_overall_accuracy().numpy()) # Correct/Total = (1+1)/(1+1+1) = 2/3 = 0.666
    print("Mean IoU (excluding background):", metric.compute_mean_iou(ignore_background=True).numpy()) # mean(0.5, 0.5) = 0.5
    print("Mean IoU (including background):", metric.compute_mean_iou(ignore_background=False).numpy()) # mean(nan, 0.5, 0.5) -> 0.5 (ignores nan)

    metric.plot_confusion_matrix(class_names=class_names_example, save_path='./result-debug/test_cm_ignore.png', normalize='true')
    metric.plot_confusion_matrix(class_names=class_names_example, save_path='./result-debug/test_cm_ignore_counts.png', normalize=None)
