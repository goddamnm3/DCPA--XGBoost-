"""
口腔健康风险预测系统 - 完整实现
该算法使用支持向量机(SVM)作为预测模型，集成了特征工程、模型优化和解释性功能

主要功能：
1. 数据预处理和特征工程
2. SVM模型训练和优化
3. 模型解释（SHAP值分析）
4. 单次和批量预测
5. 模型保存和加载
"""

# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.svm import SVC  # 支持向量机分类器
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report, f1_score)
import shap
import joblib
import warnings
import os
from collections import OrderedDict
from sklearn.base import clone
from itertools import product
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# 配置警告设置
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 常量定义
MODEL_FILE = "../dental_svm_model.pkl"  # 模型保存文件名
DATA_FILE = r"D:\dachuang\processed_data2.xlsx"  # 数据文件路径


class DentalHealthPredictor:
    def __init__(self):
        self.model = None  # 存储训练好的SVM模型
        self.imputer = None  # 缺失值处理器
        self.scaler = None  # 特征缩放器
        self.label_encoders = {}  # 分类变量编码器字典
        self.features = []  # 特征列表
        self.training_data = None  # 训练数据，用于SHAP解释
        self.best_params = None  # 存储最佳参数
        self.target_accuracy = 0.85  # 目标准确率
        self.feature_selector = None  # 特征选择器

    def load_data(self, file_path):
        try:
            data = pd.read_excel(file_path)
            print("数据已成功加载！样本数:", len(data))

            print("\n目标变量分布情况:")
            print(data['if_illness'].value_counts(normalize=True))

            cols_to_drop = ['writer_id', 'whether_cure', 'descsyed_counter']
            data = data.drop([col for col in cols_to_drop if col in data.columns], axis=1)

            data['if_illness'] = pd.to_numeric(data['if_illness'], errors='coerce')

            return data
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            raise

    def preprocess_data(self, data):
        categorical_features = [
            "gender", "brush_method", "toothpaste", "wash_meal",
            "floss_seq", "wash_seq", "sweet_seq", "sweet_drink_seq",
            "other_snack_seq", "dental_seq", "first_brush", "help_until",
            "first_check", "first_decsyed", "descsyed_cure", "mike_method",
            "whether_F", "sealant", "systemic_disease", "self_brush",
            "parent_edu", "parent_job", "parent_health", "parent_emphasis"
        ]

        print("\n正在编码分类变量...")
        for col in categorical_features:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le

        print("正在处理连续变量...")
        continuous_features = [col for col in data.columns
                               if col not in categorical_features + ['if_illness']]

        for col in continuous_features:
            if col in data.columns:
                try:
                    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                    data[col] = discretizer.fit_transform(data[[col]].values)
                except Exception as e:
                    print(f"无法分箱变量 {col}: {str(e)}")
                    continue

        print("正在创建交互特征...")
        data['oral_hygiene_score'] = (
                data['brush_method'] + data['floss_seq'] + data['wash_seq']
        )

        data['sweet_hygiene_interaction'] = data['sweet_seq'] * data['oral_hygiene_score']
        data['parent_edu_hygiene'] = data['parent_edu'] * data['oral_hygiene_score']

        print("正在创建时间相关特征...")
        if 'first_check' in data.columns and 'first_decsyed' in data.columns:
            data['check_to_decsyed_interval'] = data['first_decsyed'] - data['first_check']

        self.features = [col for col in data.columns if col != 'if_illness']
        X = data[self.features]
        y = data["if_illness"]

        print("正在处理缺失值...")
        self.imputer = SimpleImputer(strategy='most_frequent')
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)

        print("正在标准化数据...")
        self.scaler = StandardScaler()
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        return X, y

    def optimize_parameters(self, X_train, y_train, X_val, y_val):
        """
        使用GridSearchCV优化SVM参数
        """
        print("\n开始参数优化...")

        # 定义参数网格 - 针对SVM优化
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],  # 正则化参数
            'svm__kernel': ['linear', 'rbf', 'poly'],  # 核函数
            'svm__gamma': ['scale', 'auto', 0.1, 1],  # 核系数
            'svm__degree': [2, 3, 4],  # 多项式核的阶数
            'svm__class_weight': ['balanced', None],
            'svm__probability': [True]  # 必须为True以便获取概率
        }

        # 创建特征选择器和SVM的Pipeline
        pipeline = Pipeline([
            ('feature_selection', SelectKBest(f_classif)),
            ('svm', SVC(random_state=42))
        ])

        # 创建GridSearchCV对象
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,  # 5折交叉验证
            scoring='accuracy',
            n_jobs=-1,  # 使用所有CPU核心
            verbose=1
        )

        # 训练并找到最佳参数
        print("开始网格搜索...")
        grid_search.fit(X_train, y_train)

        # 获取最佳参数和模型
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_

        print("\n网格搜索完成!")
        print(f"最佳参数: {best_params}")
        print(f"最佳交叉验证分数: {best_score:.4f}")

        # 评估最佳模型在不同数据集上的表现
        train_score = accuracy_score(y_train, best_model.predict(X_train))
        val_score = accuracy_score(y_val, best_model.predict(X_val))

        print(f"\n训练集准确率: {train_score:.4f}")
        print(f"验证集准确率: {val_score:.4f}")

        # 保存特征选择器和最佳参数
        self.feature_selector = best_model.named_steps['feature_selection']
        self.best_params = best_params

        return best_model

    def train_model(self, X, y):
        """
        训练优化后的SVM模型
        """
        print("\n开始训练SVM模型...")

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 优化参数
        self.model = self.optimize_parameters(X_train, y_train, X_val, y_val)

        # 使用最佳参数重新训练完整训练集
        print("\n使用最佳参数训练最终模型...")
        self.model.fit(X, y)

        # 打印最终模型性能
        train_score = accuracy_score(y, self.model.predict(X))
        print(f"最终模型训练集准确率: {train_score:.4f}")

        # 保存训练数据用于SHAP解释
        self.training_data = X
        print("训练数据已保存，可用于SHAP解释")

        return self.model

    def evaluate_and_adjust(self, X_train, y_train, X_val, y_val):
        print("\n=== 模型评估与调整 ===")

        # 初始评估
        y_pred = self.model.predict(X_val)
        initial_score = roc_auc_score(y_val, self.model.predict_proba(X_val)[:, 1])
        print(f"初始验证集 AUC: {initial_score:.4f}")

        # 获取支持向量
        print("\n支持向量数量:", len(self.model.named_steps['svm'].support_vectors_))
        print("支持向量比例:", len(self.model.named_steps['svm'].support_vectors_) / len(X_train))

        # 识别重要特征
        try:
            if hasattr(self.model.named_steps['svm'], 'coef_'):
                coef = self.model.named_steps['svm'].coef_
                if len(coef.shape) > 1:
                    coef = coef[0]

                importance = pd.DataFrame({
                    'feature': X_train.columns[self.feature_selector.get_support()],
                    'importance': np.abs(coef)
                }).sort_values('importance', ascending=False)

                print("\n前10个最重要的特征:")
                print(importance.head(10))

                # 尝试移除低重要性特征
                improved = False
                for threshold in [0.05, 0.1, 0.15]:
                    low_imp_features = importance[
                        importance['importance'] < importance['importance'].quantile(threshold)]

                    if len(low_imp_features) == 0:
                        continue

                    print(f"\n尝试移除重要性低于 {threshold} 分位数的特征...")
                    selected_features = [f for f in X_train.columns if f not in low_imp_features['feature']]

                    # 使用精选特征重新训练
                    temp_model = clone(self.model)
                    temp_model.fit(X_train[selected_features], y_train)

                    # 评估
                    new_score = roc_auc_score(y_val, temp_model.predict_proba(X_val[selected_features])[:, 1])
                    print(f"新验证集 AUC: {new_score:.4f} (原: {initial_score:.4f})")

                    if new_score > initial_score + 0.01:  # 显著提升
                        print("模型性能提升，保留特征调整")
                        self.model = temp_model
                        self.features = selected_features
                        improved = True
                        break

                if not improved:
                    print("特征移除未带来显著改进，保留原特征集")
        except Exception as e:
            print(f"特征重要性分析时出错: {str(e)}")

        # 最终评估
        final_score = roc_auc_score(y_val, self.model.predict_proba(X_val)[:, 1])
        print(f"\n最终验证集 AUC: {final_score:.4f} (改进: {final_score - initial_score:+.4f})")

        return self.model

    def evaluate_model(self, X, y, dataset_name="测试集"):
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        print(f"\n===== {dataset_name}评估结果 =====")

        print("\n基础评估指标:")
        print(f"准确率: {accuracy_score(y, y_pred):.3f}")
        print(f"AUC分数: {roc_auc_score(y, y_proba):.3f}")
        print(f"F1分数: {f1_score(y, y_pred):.3f}")

        print("\n分类报告:")
        print(classification_report(y, y_pred))

        print("\n混淆矩阵:")
        cm = confusion_matrix(y, y_pred)
        print(cm)

        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        print(f"\n特异度: {specificity:.3f}")
        print(f"敏感度: {sensitivity:.3f}")

        print("\n预测概率分布:")
        print(pd.Series(y_proba).describe())

        print("\n特征重要性分析:")
        try:
            if hasattr(self.model.named_steps['svm'], 'coef_'):
                coef = self.model.named_steps['svm'].coef_
                if len(coef.shape) > 1:
                    coef = coef[0]

                importance = pd.DataFrame({
                    'feature': X.columns[self.feature_selector.get_support()],
                    'importance': np.abs(coef)
                }).sort_values('importance', ascending=False)

                print("\n前10个最重要的特征:")
                print(importance.head(10))

                total_importance = importance['importance'].sum()
                importance['importance_percentage'] = (importance['importance'] / total_importance * 100).round(2)

                print("\n特征重要性百分比:")
                for idx, row in importance.head(10).iterrows():
                    print(f"{row['feature']}: {row['importance_percentage']}%")
        except Exception as e:
            print(f"特征重要性分析时出错: {str(e)}")

    def save_model(self, file_path):
        model_data = {
            'model': self.model,
            'imputer': self.imputer,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'features': self.features,
            'feature_selector': self.feature_selector
        }
        joblib.dump(model_data, file_path)
        print(f"\n模型已保存到 {file_path}")

    def load_model(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件 {file_path} 不存在")

        model_data = joblib.load(file_path)
        self.model = model_data['model']
        self.imputer = model_data['imputer']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.features = model_data['features']
        self.feature_selector = model_data.get('feature_selector', None)
        print(f"\n已从 {file_path} 加载模型")

    def predict_risk(self, input_data):
        input_df = pd.DataFrame([input_data])[self.features]

        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                try:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                except:
                    print(f"警告：无法将列 {col} 转换为数值类型")

        input_df = pd.DataFrame(
            self.imputer.transform(input_df),
            columns=input_df.columns
        )

        input_df = pd.DataFrame(
            self.scaler.transform(input_df),
            columns=input_df.columns
        )

        prob = self.model.predict_proba(input_df)[0, 1]
        risk = "高风险" if prob > 0.7 else ("中风险" if prob > 0.5 else "低风险")

        explanation = "\n无法生成详细解释，但预测结果仍然有效。"

        try:
            # 使用KernelExplainer进行SHAP值分析
            def model_predict(x):
                return self.model.predict_proba(x)[:, 1]

            if hasattr(self, 'training_data'):
                background_data = shap.sample(self.training_data, 100)
            else:
                background_data = shap.sample(input_df, 100)

            explainer = shap.KernelExplainer(model_predict, background_data)
            shap_values = explainer.shap_values(input_df)

            feature_contributions = sorted(
                zip(self.features, shap_values[0]),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # 定义可控因素列表
            controllable_factors = [
                "brush_freq", "brush_time", "brush_method", "toothpaste",
                "wash_meal", "floss_seq", "wash_seq", "sweet_seq",
                "sweet_drink_seq", "other_snack_seq", "dental_seq",
                "first_brush", "help_until", "first_check", "first_decsyed",
                "descsyed_cure", "mike_method", "whether_F", "sealant",
                "self_brush"
            ]

            controllable_contributions = [
                (feature, value) for feature, value in feature_contributions
                if feature in controllable_factors
            ]

            top_controllable = controllable_contributions[:3]

            explanation = "\n主要可控影响因素:\n"
            for feature, value in top_controllable:
                direction = "增加风险" if value > 0 else "降低风险"
                explanation += f"- {feature}: {direction} ({abs(value):.3f})\n"

            explanation += "\n建议措施:\n"
            for feature, value in top_controllable:
                if value > 0:
                    if feature == "brush_freq":
                        explanation += f"- 增加刷牙频率，建议每天至少刷牙2-3次\n"
                    elif feature == "brush_time":
                        explanation += f"- 延长刷牙时间，建议每次刷牙至少2分钟\n"
                    elif feature == "brush_method":
                        explanation += f"- 改进刷牙方法，建议使用巴氏刷牙法\n"
                    elif feature == "toothpaste":
                        explanation += f"- 使用含氟牙膏，有助于预防龋齿\n"
                    elif feature == "wash_meal":
                        explanation += f"- 饭后及时漱口，减少食物残渣\n"
                    elif feature == "floss_seq":
                        explanation += f"- 增加使用牙线的频率，清洁牙缝\n"
                    elif feature == "wash_seq":
                        explanation += f"- 增加漱口频率，保持口腔清洁\n"
                    elif feature == "sweet_seq":
                        explanation += f"- 减少甜食摄入频率，控制糖分摄入\n"
                    elif feature == "sweet_drink_seq":
                        explanation += f"- 减少甜饮料摄入频率，避免含糖饮料\n"
                    elif feature == "other_snack_seq":
                        explanation += f"- 减少零食摄入频率，特别是粘性零食\n"
                    elif feature == "dental_seq":
                        explanation += f"- 增加看牙医的频率，定期口腔检查\n"
                    elif feature == "first_brush":
                        explanation += f"- 尽早开始刷牙习惯，从小培养口腔卫生意识\n"
                    elif feature == "help_until":
                        explanation += f"- 延长父母协助刷牙的时间，确保刷牙质量\n"
                    elif feature == "first_check":
                        explanation += f"- 尽早进行口腔检查，及早发现问题\n"
                    elif feature == "first_decsyed":
                        explanation += f"- 及时处理龋齿问题，避免病情加重\n"
                    elif feature == "descsyed_cure":
                        explanation += f"- 及时治疗龋齿，避免并发症\n"
                    elif feature == "mike_method":
                        explanation += f"- 改进口腔清洁方法，使用正确的清洁工具\n"
                    elif feature == "whether_F":
                        explanation += f"- 考虑使用含氟产品，增强牙齿抗龋能力\n"
                    elif feature == "sealant":
                        explanation += f"- 考虑进行窝沟封闭，预防窝沟龋\n"
                    elif feature == "self_brush":
                        explanation += f"- 培养自主刷牙习惯，提高口腔卫生意识\n"
                    else:
                        explanation += f"- 改善{feature}相关习惯，降低龋齿风险\n"

            explanation += "\n特征重要性分析:\n"
            try:
                if hasattr(self.model.named_steps['svm'], 'coef_'):
                    coef = self.model.named_steps['svm'].coef_
                    if len(coef.shape) > 1:
                        coef = coef[0]

                    importance = pd.DataFrame({
                        'feature': self.features[self.feature_selector.get_support()],
                        'importance': np.abs(coef)
                    }).sort_values('importance', ascending=False)

                    total_importance = importance['importance'].sum()
                    importance['importance_percentage'] = (importance['importance'] / total_importance * 100).round(2)

                    for idx, row in importance.head(5).iterrows():
                        explanation += f"- {row['feature']}: {row['importance_percentage']}%\n"
            except Exception as e:
                print(f"特征重要性分析时出错: {str(e)}")

            explanation += "\n风险因素分析:\n"
            risk_increasing = [f for f, v in feature_contributions if v > 0]
            risk_decreasing = [f for f, v in feature_contributions if v < 0]

            if risk_increasing:
                explanation += "增加风险的因素:\n"
                for feature in risk_increasing:
                    explanation += f"- {feature}\n"

            if risk_decreasing:
                explanation += "降低风险的因素:\n"
                for feature in risk_decreasing:
                    explanation += f"- {feature}\n"

        except Exception as e:
            print(f"生成解释时出错: {str(e)}")

        return {
            'probability': prob,
            'risk_level': risk,
            'explanation': explanation
        }


def main():
    """
    主程序入口

    功能：
    1. 初始化预测器
    2. 检查是否存在已训练的模型
    3. 提供交互式菜单进行预测操作

    菜单选项：
    1. 单次预测：对单个样本进行预测
    2. 批量预测：对多个样本进行预测
    3. 退出：结束程序
    """
    print("===== 口腔健康风险预测系统 (SVM版本) =====")
    predictor = DentalHealthPredictor()  # 创建预测器实例

    # 检查是否已有训练好的模型
    if os.path.exists(MODEL_FILE):
        choice = input("检测到已有模型，是否重新训练？(y/n): ").lower()
        if choice == 'y':
            train_new_model(predictor)  # 训练新模型
        else:
            try:
                predictor.load_model(MODEL_FILE)  # 加载已有模型
            except Exception as e:
                print(f"加载模型时出错: {str(e)}")
                print("将重新训练模型...")
                train_new_model(predictor)
    else:
        print("未检测到已有模型，将进行训练...")
        train_new_model(predictor)  # 训练新模型

    # 进入预测循环
    while True:
        print("\n===== 预测菜单 =====")
        print("1. 单次预测")
        print("2. 批量预测")
        print("3. 退出")

        choice = input("请选择操作: ").strip()

        if choice == '1':
            single_prediction(predictor)  # 单次预测
        elif choice == '2':
            batch_prediction(predictor)  # 批量预测
        elif choice == '3':
            print("退出系统...")
            break
        else:
            print("无效输入，请重新选择")


def train_new_model(predictor):
    """
    训练新模型流程
    """
    print("\n===== 模型训练 =====")
    try:
        # 1. 加载数据
        data = predictor.load_data(DATA_FILE)

        # 2. 预处理
        X, y = predictor.preprocess_data(data)

        # 3. 划分训练集、验证集和测试集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        print("\n数据集划分情况:")
        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        print(f"测试集大小: {len(X_test)}")

        # 4. 训练模型
        predictor.model = predictor.train_model(X_train, y_train)

        # 5. 评估模型
        print("\n开始模型评估与调整...")
        predictor.evaluate_and_adjust(X_train, y_train, X_val, y_val)

        print("\n训练集评估结果:")
        predictor.evaluate_model(X_train, y_train, "训练集")

        print("\n验证集评估结果:")
        predictor.evaluate_model(X_val, y_val, "验证集")

        print("\n测试集评估结果:")
        predictor.evaluate_model(X_test, y_test, "测试集")

        # 6. 检查是否满足目标准确率
        train_acc = accuracy_score(y_train, predictor.model.predict(X_train))
        val_acc = accuracy_score(y_val, predictor.model.predict(X_val))
        test_acc = accuracy_score(y_test, predictor.model.predict(X_test))

        if (train_acc >= predictor.target_accuracy and
                val_acc >= predictor.target_accuracy and
                test_acc >= predictor.target_accuracy):
            print("\n模型性能满足要求，保存模型...")
            predictor.save_model(MODEL_FILE)
        else:
            print("\n警告：模型性能未达到目标准确率")
            choice = input("是否仍要保存模型？(y/n): ").lower()
            if choice == 'y':
                predictor.save_model(MODEL_FILE)
            else:
                print("模型未保存，请重新训练或调整目标准确率")

    except Exception as e:
        print(f"训练过程中出错: {str(e)}")


def single_prediction(predictor):
    """
    单次预测流程

    参数：
    predictor: DentalHealthPredictor实例

    功能：
    1. 检查模型是否已加载
    2. 收集用户输入的特征值
    3. 处理缺失值（使用默认值）
    4. 进行预测并显示结果
    5. 显示预测解释
    """
    print("\n===== 单次预测 =====")

    # 检查模型是否已加载
    if predictor.model is None or predictor.imputer is None:
        print("错误：模型未加载，请先训练或加载模型")
        return

    input_data = {}  # 存储用户输入

    # 获取用户输入
    for feature in predictor.features:
        while True:
            value = input(f"{feature} (留空使用默认值): ").strip()

            if value == "":
                # 使用默认值(分类变量使用最常见的类别，数值变量使用0)
                if feature in predictor.label_encoders:
                    default = predictor.label_encoders[feature].classes_[0]
                else:
                    default = 0
                input_data[feature] = default
                break

            try:
                # 处理分类变量
                if feature in predictor.label_encoders:
                    if value in predictor.label_encoders[feature].classes_:
                        input_data[feature] = predictor.label_encoders[feature].transform([value])[0]
                    else:
                        print(f"无效值，可选: {list(predictor.label_encoders[feature].classes_)}")
                        continue
                else:
                    # 处理数值变量
                    input_data[feature] = float(value)
                break
            except ValueError:
                print("请输入有效值")
                continue

    # 进行预测
    try:
        result = predictor.predict_risk(input_data)
        print(f"\n预测结果: 患病概率 = {result['probability']:.2%}")
        print(f"风险等级: {result['risk_level']}")
        print(result['explanation'])  # 打印解释
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")


def batch_prediction(predictor):
    """
    批量预测流程

    参数：
    predictor: DentalHealthPredictor实例

    功能：
    1. 读取批量数据文件
    2. 预处理数据
    3. 进行批量预测
    4. 保存预测结果

    输入文件格式：
    - CSV或Excel文件
    - 必须包含所有必要的特征列
    """
    print("\n===== 批量预测 =====")
    file_path = input("请输入数据文件路径(CSV/Excel): ").strip()

    try:
        # 读取数据文件
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            data = pd.read_excel(file_path)

        # 预处理 - 使用保存的imputer处理缺失值
        processed_data = pd.DataFrame(
            predictor.imputer.transform(data[predictor.features]),
            columns=predictor.features
        )

        # 标准化数据
        processed_data = pd.DataFrame(
            predictor.scaler.transform(processed_data),
            columns=predictor.features
        )

        # 预测 - 获取预测概率
        probabilities = predictor.model.predict_proba(processed_data)[:, 1]
        data['预测概率'] = probabilities
        # 根据概率划分风险等级
        data['风险等级'] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1],
            labels=['低风险', '中风险', '高风险']
        )

        # 保存结果 - 在原文件名后添加"_预测结果"
        output_path = os.path.splitext(file_path)[0] + "_预测结果.xlsx"
        data.to_excel(output_path, index=False)
        print(f"\n预测完成！结果已保存到 {output_path}")

    except Exception as e:
        print(f"批量预测过程中出错: {str(e)}")


if __name__ == "__main__":
    main()  # 启动主程序