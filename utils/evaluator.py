import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, f1_score, precision_score,
    recall_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def evaluate_model(predictor, X, y, dataset_name="测试集"):
    """
    评估模型性能
    
    参数：
    predictor: 预测器实例
    X: 特征数据
    y: 目标变量
    dataset_name: 数据集名称
    """
    try:
        # 获取预测结果
        y_pred = predictor.model.predict(X)
        y_pred_proba = predictor.model.predict_proba(X)[:, 1]
        
        # 计算评估指标
        accuracy = accuracy_score(y, y_pred)
        auc_roc = roc_auc_score(y, y_pred_proba)
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)  # 特异度
        sensitivity = tp / (tp + fn)  # 敏感度
        
        # 打印评估结果
        print(f"\n===== {dataset_name}评估结果 =====")
        print(f"准确率: {accuracy:.4f}")
        print(f"AUC-ROC分数: {auc_roc:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"特异度: {specificity:.4f}")
        print(f"敏感度: {sensitivity:.4f}")
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y, y_pred))
        
        # 打印混淆矩阵
        print("\n混淆矩阵:")
        print(cm)
        
        # 绘制ROC曲线
        plot_roc_curve(y, y_pred_proba, dataset_name)
        
        # 绘制校准曲线
        plot_calibration_curve(y, y_pred_proba, dataset_name)
        
        return {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'sensitivity': sensitivity
        }
        
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        raise

def plot_roc_curve(y_true, y_pred_proba, dataset_name):
    """
    绘制ROC曲线
    
    参数：
    y_true: 真实标签
    y_pred_proba: 预测概率
    dataset_name: 数据集名称
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title(f'{dataset_name} ROC曲线')
        plt.legend(loc="lower right")
        plt.show()
        
    except Exception as e:
        print(f"绘制ROC曲线时出错: {str(e)}")

def plot_calibration_curve(y_true, y_pred_proba, dataset_name):
    """
    绘制校准曲线
    
    参数：
    y_true: 真实标签
    y_pred_proba: 预测概率
    dataset_name: 数据集名称
    """
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, 's-', label=f'{dataset_name}')
        plt.plot([0, 1], [0, 1], '--', label='完美校准')
        plt.xlabel('预测概率')
        plt.ylabel('实际概率')
        plt.title(f'{dataset_name} 校准曲线')
        plt.legend()
        plt.show()
        
    except Exception as e:
        print(f"绘制校准曲线时出错: {str(e)}")

def plot_feature_importance(predictor, X):
    """
    绘制特征重要性图
    
    参数：
    predictor: 预测器实例
    X: 特征数据
    """
    try:
        if hasattr(predictor.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': predictor.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=importance.head(20))
            plt.title('前20个重要特征')
            plt.xlabel('重要性')
            plt.ylabel('特征')
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"绘制特征重要性图时出错: {str(e)}") 