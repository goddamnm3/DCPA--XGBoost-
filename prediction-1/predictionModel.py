"""
口腔健康风险预测系统 - 完整实现
该算法使用XGBoost作为预测模型，集成了特征工程、模型优化、校准和解释性功能

主要功能：
1. 数据预处理和特征工程
2. XGBoost模型训练和优化
3. 贝叶斯优化超参数
4. 概率校准
5. 模型解释（SHAP值分析）
6. 单次和批量预测
7. 模型保存和加载
"""

# 导入必要的库
import pandas as pd  # 数据处理和分析，用于数据框操作
import numpy as np  # 数值计算，用于数组操作和数学计算
import xgboost as xgb  # XGBoost算法，用于梯度提升树模型
from sklearn.model_selection import train_test_split, StratifiedKFold  # 数据分割和交叉验证
from sklearn.impute import SimpleImputer  # 缺失值处理，使用简单填充策略
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer  # 数据预处理，包括标签编码和分箱
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report, f1_score)  # 模型评估指标
from sklearn.calibration import CalibratedClassifierCV  # 概率校准，使预测概率更准确
from skopt import BayesSearchCV  # 贝叶斯优化，用于超参数搜索
from skopt.space import Real, Integer, Categorical  # 参数空间定义，用于贝叶斯优化
import shap  # 模型解释，用于分析特征重要性
import joblib  # 模型保存和加载，用于持久化模型
import warnings  # 警告处理，用于控制警告信息
import os  # 操作系统功能，用于文件操作
from collections import OrderedDict  # 有序字典，用于保持特征顺序
from functools import partial  # 函数工具，用于创建偏函数
from sklearn.inspection import partial_dependence  # 部分依赖图，用于分析特征影响
from sklearn.base import clone  # 模型克隆，用于创建模型的深拷贝
import matplotlib.pyplot as plt  # 用于绘制图形

# 配置警告设置 - 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)  # 忽略用户警告
warnings.filterwarnings("ignore", category=FutureWarning)  # 忽略未来版本警告

# 常量定义
MODEL_FILE = "advanced_dental_model.pkl"  # 模型保存文件名
DATA_FILE = r"D:\dachuang\processed_data1.xlsx" # 数据文件路径


class DentalHealthPredictor:
    """
    口腔健康预测系统主类
    包含数据预处理、特征工程、模型训练和预测功能

    主要组件：
    1. model: 训练好的XGBoost模型
    2. imputer: 缺失值处理器
    3. label_encoders: 分类变量编码器字典
    4. features: 特征列表
    5. calibrator: 概率校准器
    6. shap_explainer: SHAP值解释器
    """

    def __init__(self):
        """
        初始化预测器
        设置所有必要的组件为None，这些组件将在后续步骤中被填充
        """
        self.model = None  # 存储训练好的XGBoost模型
        self.imputer = None  # 缺失值处理器，用于处理数据中的缺失值
        self.label_encoders = {}  # 分类变量编码器字典，用于存储每个分类变量的编码器
        self.features = []  # 特征列表，存储所有用于预测的特征名称
        self.calibrator = None  # 概率校准器，用于校准模型预测的概率
        self.shap_explainer = None  # SHAP解释器，用于分析特征重要性
        self.training_data = None  # 训练数据
        self.target_variable = None  # 目标变量
        self.original_features = []  # 存储所有原始特征

    def load_data(self, file_path):
        """
        加载并预处理数据

        参数：
        file_path: 数据文件路径（Excel格式）

        返回：
        DataFrame: 处理后的数据框

        处理步骤：
        1. 读取Excel文件
        2. 检查目标变量分布
        3. 移除无关列
        4. 确保目标变量为数值型
        """
        try:
            # 从Excel文件读取数据
            data = pd.read_excel(file_path)
            print("数据已成功加载！样本数:", len(data))

            # 检查目标变量分布
            print("\n目标变量分布情况:")
            print(data['if_illness'].value_counts(normalize=True))

            # 数据清理 - 移除无关列
            cols_to_drop = ['writer_id', 'descsyed_counter']
            data = data.drop([col for col in cols_to_drop if col in data.columns], axis=1)

            # 确保目标变量是数值型
            data['if_illness'] = pd.to_numeric(data['if_illness'], errors='coerce')

            return data
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            raise

    def preprocess_data(self, data):
        """
        数据预处理和特征工程

        参数：
        data: 原始数据DataFrame

        返回：
        tuple: (X, y) 处理后的特征矩阵和目标变量

        处理步骤：
        1. 分类变量编码
        2. 连续变量分箱
        3. 特征选择
        4. 处理缺失值
        """
        # 定义分类变量列表
        categorical_features = [
            "gender", "brush_method", "toothpaste", "wash_meal",
            "floss_seq", "wash_seq", "sweet_seq", "sweet_drink_seq",
            "other_snack_seq", "dental_seq", "first_brush", "help_until",
            "first_check", "first_descsyed", "descsyed_cure", "mike_method",
            "whether_F", "sealant", "systemic_disease", "self_brush",
            "parent_edu", "parent_job", "parent_health", "parent_emphasis",
            "whether_cure"
        ]

        # 1. 分类变量编码 - 使用LabelEncoder将分类变量转换为数字
        print("\n正在编码分类变量...")
        for col in categorical_features:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le  # 保存编码器供后续使用

        # 2. 连续变量分箱 - 将连续变量离散化为有序类别
        print("正在处理连续变量...")
        continuous_features = [col for col in data.columns
                               if col not in categorical_features + ['if_illness']]

        for col in continuous_features:
            if col in data.columns:
                try:
                    # 使用等频分箱(每个箱中样本数量相同)
                    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                    data[col] = discretizer.fit_transform(data[[col]].values)
                except Exception as e:
                    print(f"无法分箱变量 {col}: {str(e)}")
                    continue

        # 3. 特征选择 - 排除目标变量
        self.features = [col for col in data.columns if col != 'if_illness']
        X = data[self.features]
        y = data["if_illness"]

        # 4. 处理缺失值 - 使用最常见的值填充缺失值
        print("正在处理缺失值...")
        self.imputer = SimpleImputer(strategy='most_frequent')
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)

        self.original_features = self.features.copy()  # 保存原始特征

        return X, y

    def train_model(self, X, y):
        """
        训练优化后的XGBoost模型
        """
        # 保存训练数据和目标变量
        self.training_data = X.copy()
        self.target_variable = y.copy()

        try:
            # 1. 使用贝叶斯优化搜索最佳参数
            print("\n开始贝叶斯优化参数搜索...")

            # 定义参数空间 - 调整参数范围以提高精确度
            param_space = {
                'learning_rate': Real(0.01, 0.03),  # 缩小学习率范围，避免过拟合
                'max_depth': Integer(3, 6),  # 降低树的最大深度，减少模型复杂度
                'n_estimators': Integer(100, 300),  # 增加树的数量，但使用较小的学习率
                'subsample': Real(0.6, 0.8),  # 降低子采样比例，增加随机性
                'colsample_bytree': Real(0.6, 0.8),  # 降低特征采样比例，增加随机性
                'min_child_weight': Integer(3, 7),  # 增加最小叶子节点权重，减少过拟合
                'gamma': Real(0.2, 1.0),  # 调整gamma值范围
                'reg_alpha': Real(0.5, 2.0),  # 增加L1正则化
                'reg_lambda': Real(1.0, 3.0)  # 增加L2正则化
            }

            # 创建基础模型
            base_model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_jobs=-1,
                tree_method='hist',
                device='cpu',  # 强制使用CPU
                max_bin=128,  # 减少分箱数，降低模型复杂度
                scale_pos_weight=1.0,
                max_leaves=32  # 限制叶子节点数量，减少模型复杂度
            )

            # 使用贝叶斯优化
            bayes_search = BayesSearchCV(
                estimator=base_model,
                search_spaces=param_space,
                n_iter=150,  # 增加迭代次数以找到更好的参数
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )

            # 执行参数搜索
            bayes_search.fit(X, y)

            # 获取最佳参数
            best_params = bayes_search.best_params_
            print("\n找到的最佳参数组合:")
            for param, value in best_params.items():
                print(f"{param}: {value}")

            # 2. 使用最佳参数创建最终模型
            final_params = {
                'objective': 'binary:logistic',
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist',
                'device': 'cpu',  # 使用device参数替代gpu_id
                'max_bin': 128,
                'scale_pos_weight': 1.0,  # 添加类别平衡参数
                **best_params
            }

            print("\n开始训练最终模型...")
            self.model = xgb.XGBClassifier(**final_params)
            self.model.fit(X, y)
            print("模型训练完成")

            # 3. 保存特征重要性
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)

            return self.model

        except Exception as e:
            print(f"训练过程中出错: {str(e)}")
            # 尝试使用更简单的参数重新训练
            try:
                print("\n尝试使用更简单的参数重新训练...")
                simple_params = {
                    'objective': 'binary:logistic',
                    'max_depth': 5,
                    'n_estimators': 100,  # 增加树的数量
                    'learning_rate': 0.03,  # 降低学习率
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 3,  # 增加最小叶子节点权重
                    'gamma': 0.3,  # 增加gamma值
                    'reg_alpha': 0.2,  # 增加L1正则化
                    'reg_lambda': 1.0,  # 增加L2正则化
                    'random_state': 42,
                    'tree_method': 'hist',
                    'device': 'cpu',  # 使用device参数替代gpu_id
                    'scale_pos_weight': 1.0  # 添加类别平衡参数
                }
                self.model = xgb.XGBClassifier(**simple_params)
                self.model.fit(X, y)
                print("使用简单参数训练成功")
                return self.model
            except Exception as e2:
                print(f"使用简单参数训练也失败: {str(e2)}")
                raise

    def evaluate_and_adjust(self, X_train, y_train, X_val, y_val):
        """
        新增方法：基于验证集性能调整模型
        """
        print("\n=== 模型评估与调整 ===")

        # 初始评估
        y_pred = self.model.predict(X_val)
        initial_score = roc_auc_score(y_val, self.model.predict_proba(X_val)[:, 1])
        print(f"初始验证集 AUC: {initial_score:.4f}")

        # 获取原始XGBoost模型
        if hasattr(self.model, 'base_estimator_'):
            base_model = self.model.base_estimator_
        else:
            base_model = self.model

        # 获取特征重要性
        try:
            if hasattr(base_model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': base_model.feature_importances_
                }).sort_values('importance', ascending=False)

                # 识别表现不佳的特征
                poor_features = importance[importance['importance'] < importance['importance'].quantile(0.1)]['feature']

                if len(poor_features) > 0:
                    print(f"\n检测到 {len(poor_features)} 个低重要性特征:")
                    print(poor_features.tolist())

                    # 尝试移除低重要性特征
                    improved = False
                    for threshold in [0.05, 0.1, 0.15]:
                        low_imp_features = importance[importance['importance'] < importance['importance'].quantile(threshold)]

                        if len(low_imp_features) == 0:
                            continue

                        print(f"\n尝试移除重要性低于 {threshold} 分位数的特征...")
                        selected_features = [f for f in X_train.columns if f not in low_imp_features['feature']]

                        # 使用精选特征重新训练
                        temp_model = clone(base_model)
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
            else:
                print("无法获取特征重要性信息")
        except Exception as e:
            print(f"特征重要性分析时出错: {str(e)}")

        # 最终评估
        final_score = roc_auc_score(y_val, self.model.predict_proba(X_val[self.features])[:, 1])
        print(f"\n最终验证集 AUC: {final_score:.4f} (改进: {final_score - initial_score:+.4f})")

        return self.model

    def evaluate_model(self, X, y, dataset_name="测试集"):
        """
        评估模型性能

        参数：
        X: 特征数据
        y: 目标变量
        dataset_name: 数据集名称（训练集/验证集/测试集）
        """
        # 获取预测结果
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # 计算准确率
        accuracy = accuracy_score(y, y_pred)

        # 计算AUC-ROC分数
        auc_roc = roc_auc_score(y, y_pred_proba)

        # 计算F1分数
        f1 = f1_score(y, y_pred)

        # 计算混淆矩阵
        cm = confusion_matrix(y, y_pred)

        # 计算特异度和敏感度
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)  # 特异度（真阴性率）
        sensitivity = tp / (tp + fn)  # 敏感度（真阳性率）

        # 打印评估结果
        print(f"\n===== {dataset_name}评估结果 =====")
        print(f"准确率: {accuracy:.4f}")
        print(f"AUC-ROC分数: {auc_roc:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"特异度: {specificity:.4f}")
        print(f"敏感度: {sensitivity:.4f}")

        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y, y_pred))

        # 打印混淆矩阵
        print("\n混淆矩阵:")
        print(cm)

        return {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'f1': f1,
            'specificity': specificity,
            'sensitivity': sensitivity
        }

    def save_model(self, file_path):
        """
        保存模型到文件

        参数：
        file_path: 模型保存的文件路径

        保存内容：
        1. model: 训练好的XGBoost模型
        2. imputer: 缺失值处理器
        3. label_encoders: 分类变量编码器字典
        4. features: 特征列表
        5. calibrator: 概率校准器

        使用joblib进行序列化，可以保存所有必要的模型组件
        """
        # 打包所有需要保存的对象
        model_data = {
            'model': self.model,
            'imputer': self.imputer,
            'label_encoders': self.label_encoders,
            'features': self.features,
            'calibrator': self.calibrator
        }
        # 使用joblib保存
        joblib.dump(model_data, file_path)
        print(f"\n模型已保存到 {file_path}")

    def load_model(self, file_path):
        """
        从文件加载模型

        参数：
        file_path: 模型文件路径

        加载内容：
        1. model: 训练好的XGBoost模型
        2. imputer: 缺失值处理器
        3. label_encoders: 分类变量编码器字典
        4. features: 特征列表
        5. calibrator: 概率校准器

        如果文件不存在，抛出FileNotFoundError异常
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件 {file_path} 不存在")

        # 加载保存的模型数据
        model_data = joblib.load(file_path)
        self.model = model_data['model']
        self.imputer = model_data['imputer']
        self.label_encoders = model_data['label_encoders']
        self.features = model_data['features']
        self.calibrator = model_data.get('calibrator', None)
        print(f"\n已从 {file_path} 加载模型")

    def generate_explanation(self, input_df, probability):
        """生成预测解释"""
        explanation = f"根据您的输入数据，系统预测您患龋齿的概率为{probability:.2%}。\n\n"

        # 获取前三个最重要的特征
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            top_features = importance.head(3)

            explanation += "对您风险贡献最大的三个因素是：\n"
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                feature = row['feature']
                value = input_df[feature].iloc[0]
                explanation += f"{i}. {feature}: {value}\n"

        # 添加建议
        explanation += "\n建议：\n"
        if probability > 0.7:
            explanation += "- 您属于高风险人群，建议尽快进行口腔检查\n"
            explanation += "- 加强口腔卫生习惯，每天至少刷牙两次，使用牙线\n"
            explanation += "- 减少糖分摄入，避免频繁食用甜食和含糖饮料\n"
            explanation += "- 定期进行专业口腔清洁和检查\n"
        elif probability > 0.3:
            explanation += "- 您属于中风险人群，需要注意口腔健康\n"
            explanation += "- 保持良好的口腔卫生习惯，定期更换牙刷\n"
            explanation += "- 控制糖分摄入，避免在睡前食用甜食\n"
            explanation += "- 每半年进行一次口腔检查\n"
        else:
            explanation += "- 您属于低风险人群，继续保持良好的口腔卫生习惯\n"
            explanation += "- 定期刷牙，使用牙线，保持口腔清洁\n"
            explanation += "- 均衡饮食，减少糖分摄入\n"
            explanation += "- 每年进行一次口腔检查\n"

        return explanation

    def predict_risk(self, input_data):
        """
        预测口腔健康风险
        """
        try:
            # 1. 准备输入数据
            input_df = pd.DataFrame([input_data])

            # 2. 确保所有特征都是数值类型
            for feature in input_df.columns:
                if feature in self.label_encoders:
                    # 如果是分类变量，确保已经转换为数值
                    if not pd.api.types.is_numeric_dtype(input_df[feature]):
                        input_df[feature] = input_df[feature].astype(float)
                else:
                    # 如果是连续变量，确保是浮点数
                    input_df[feature] = input_df[feature].astype(float)

            # 3. 创建交互特征（如果需要的特征存在）
            if all(f in input_df.columns for f in ['brush_method', 'floss_seq', 'wash_seq']):
                input_df['oral_hygiene_score'] = (
                    input_df['brush_method'].astype(float) +
                    input_df['floss_seq'].astype(float) +
                    input_df['wash_seq'].astype(float)
                )

            if 'sweet_seq' in input_df.columns and 'oral_hygiene_score' in input_df.columns:
                input_df['sweet_hygiene_interaction'] = (
                    input_df['sweet_seq'].astype(float) *
                    input_df['oral_hygiene_score'].astype(float)
                )

            if 'parent_edu' in input_df.columns and 'oral_hygiene_score' in input_df.columns:
                input_df['parent_edu_hygiene'] = (
                    input_df['parent_edu'].astype(float) *
                    input_df['oral_hygiene_score'].astype(float)
                )

            if all(f in input_df.columns for f in ['first_check', 'first_descsyed']):
                input_df['check_to_decsyed_interval'] = (
                    input_df['first_descsyed'].astype(float) -
                    input_df['first_check'].astype(float)
                )

            # 4. 确保所有模型需要的特征都存在
            missing_features = set(self.features) - set(input_df.columns)
            if missing_features:
                print(f"警告：以下特征在输入数据中缺失，将使用默认值：{missing_features}")
                for feature in missing_features:
                    if feature in self.label_encoders:
                        # 对于分类变量，使用最常见的类别
                        input_df[feature] = float(self.label_encoders[feature].transform([self.label_encoders[feature].classes_[0]])[0])
                    else:
                        # 对于连续变量，使用0
                        input_df[feature] = 0.0

            # 5. 只保留模型需要的特征
            input_df = input_df[self.features]

            # 6. 处理缺失值
            input_df = pd.DataFrame(
                self.imputer.transform(input_df),
                columns=input_df.columns
            )

            # 7. 预测概率
            prob = self.model.predict_proba(input_df)[0, 1]

            # 8. 根据新的概率区间划分风险等级
            if prob <= 0.3:  # 低风险阈值
                risk = "低风险"
            elif prob <= 0.7:  # 中风险阈值
                risk = "中风险"
            else:
                risk = "高风险"

            # 9. 计算SHAP值
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(input_df)

                # 获取特征重要性排序
                feature_importance = pd.DataFrame({
                    '特征': self.features,
                    'SHAP值': np.abs(shap_values[0])
                }).sort_values('SHAP值', ascending=False)

                # 获取前3个最重要的特征
                top_3_features = feature_importance.head(3)

                # 生成通用建议
                general_suggestions = []
                if risk == "高风险":
                    general_suggestions.extend([
                        "建议尽快预约专业牙医进行全面检查",
                        "建议加强日常口腔护理，包括正确刷牙、使用牙线和漱口水",
                        "建议控制糖分摄入，避免频繁食用甜食",
                        "建议每3个月进行一次口腔检查",
                        "建议使用含氟牙膏和漱口水"
                    ])
                elif risk == "中风险":
                    general_suggestions.extend([
                        "建议每6个月进行一次口腔检查",
                        "建议保持良好的口腔卫生习惯",
                        "建议适当控制甜食摄入",
                        "建议使用牙线清洁牙缝",
                        "建议定期更换牙刷"
                    ])
                else:
                    general_suggestions.extend([
                        "建议继续保持良好的口腔卫生习惯",
                        "建议每年进行一次口腔检查",
                        "建议保持均衡饮食，减少糖分摄入",
                        "建议定期使用牙线清洁牙缝",
                        "建议使用含氟牙膏"
                    ])

                # 生成针对性建议
                specific_suggestions = []
                for _, row in top_3_features.iterrows():
                    feature = row['特征']
                    value = input_data.get(feature, 0)

                    # 根据特征生成针对性建议
                    if feature == 'gender':
                        specific_suggestions.append(f"性别特征值为{value}，建议：男女孩口腔护理原则相同，但需关注女孩换牙期（通常早于男孩1-2年），在6-8岁第一恒磨牙萌出时加强窝沟封闭。")
                    elif feature == 'brush_method':
                        specific_suggestions.append(f"刷牙方法特征值为{value}，建议：推荐圆弧刷牙法（Fones法）：上下牙咬合，牙刷在牙面画小圈，覆盖牙龈和牙冠，每日2次，每次2分钟。")
                    elif feature == 'toothpaste':
                        specific_suggestions.append(f"牙膏使用特征值为{value}，建议：3岁以下用米粒大小（约0.1g）含氟牙膏，3岁以上用豌豆大小（0.25g），避免吞咽。")
                    elif feature == 'wash_meal':
                        specific_suggestions.append(f"餐后漱口特征值为{value}，建议：餐后用清水或儿童含氟漱口水（6岁以上）漱口30秒，减少食物残渣和酸性环境。")
                    elif feature == 'floss_seq':
                        specific_suggestions.append(f"牙线使用特征值为{value}，建议：从两颗相邻乳牙萌出后开始，家长每日用儿童牙线清洁牙缝，避免邻面龋。")
                    elif feature == 'wash_seq':
                        specific_suggestions.append(f"漱口频率特征值为{value}，建议：高龋风险儿童（如正畸治疗中）每日使用0.05%氟化钠漱口水1次，每次10ml含漱1分钟后吐出，6岁以下慎用。")
                    elif feature == 'sweet_seq':
                        specific_suggestions.append(f"甜食摄入特征值为{value}，建议：限制甜食频率，避免粘性糖果，建议随正餐食用，减少糖分暴露时间。")
                    elif feature == 'sweet_drink_seq':
                        specific_suggestions.append(f"含糖饮料摄入特征值为{value}，建议：避免含糖饮料（如碳酸饮料），饮用时使用吸管减少牙面接触，每日摄入量不超过120ml。")
                    elif feature == 'other_snack_seq':
                        specific_suggestions.append(f"其他零食摄入特征值为{value}，建议：选择高纤维零食（如苹果、胡萝卜）刺激唾液分泌，避免淀粉类零食（如饼干）残留牙面，夜间禁食含糖食物。")
                    elif feature == 'dental_seq':
                        specific_suggestions.append(f"看牙医频率特征值为{value}，建议：低龋风险儿童每6个月检查1次，高龋风险者每3个月检查+局部涂氟。")
                    elif feature == 'first_brush':
                        specific_suggestions.append(f"首次刷牙年龄特征值为{value}，建议：当婴儿长出第一颗牙齿时（通常6个月左右），即可用湿纱布轻轻擦拭牙面，预防奶瓶龋。")
                    elif feature == 'help_until':
                        specific_suggestions.append(f"需要帮助刷牙年龄特征值为{value}，建议：6岁前，家长应协助孩子刷牙，确保清洁彻底。6岁后，可逐步培养自主刷牙能力。")
                    elif feature == 'first_check':
                        specific_suggestions.append(f"首次检查年龄特征值为{value}，建议：美国儿牙协会（AAPD）建议首次牙科检查不晚于12月龄，评估唇系带、咬合及喂养方式。")
                    elif feature == 'first_descsyed':
                        specific_suggestions.append(f"首次龋齿年龄特征值为{value}，建议：如果孩子在3岁前出现龋齿，需高度重视，可能与喂养方式或口腔卫生习惯有关，应立即就医。")
                    elif feature == 'descsyed_cure':
                        specific_suggestions.append(f"龋齿治疗特征值为{value}，建议：一旦发现龋齿，应尽快治疗，避免龋齿发展为牙髓炎或根尖炎。乳牙龋齿同样需要治疗，以免影响恒牙萌出。")
                    elif feature == 'mike_method':
                        specific_suggestions.append(f"奶瓶喂养方法特征值为{value}，建议：避免让孩子含奶瓶入睡，防止奶瓶龋。1岁后逐渐过渡到杯子饮水。")
                    elif feature == 'whether_F':
                        specific_suggestions.append(f"是否使用含氟牙膏特征值为{value}，建议：推荐使用含氟牙膏，氟化物可增强牙釉质抗酸性，预防龋齿。3岁以下使用米粒大小，3岁以上使用豌豆大小。")
                    elif feature == 'sealant':
                        specific_suggestions.append(f"窝沟封闭特征值为{value}，建议：建议在6-12岁期间为恒磨牙进行窝沟封闭，预防咬合面龋齿。")
                    elif feature == 'systemic_disease':
                        specific_suggestions.append(f"全身性疾病特征值为{value}，建议：某些全身性疾病（如哮喘、糖尿病）可能影响口腔健康。定期口腔检查，必要时咨询儿科医生和牙医。")
                    elif feature == 'self_brush':
                        specific_suggestions.append(f"自主刷牙特征值为{value}，建议：6岁后逐步培养自主刷牙能力，但仍需家长监督，确保刷牙时间和方法正确。")
                    elif feature == 'parent_edu':
                        specific_suggestions.append(f"父母教育程度特征值为{value}，建议：建议家长学习正确的口腔护理知识，以身作则。")
                    elif feature == 'parent_job':
                        specific_suggestions.append(f"父母职业特征值为{value}，建议：无")
                    elif feature == 'parent_health':
                        specific_suggestions.append(f"父母口腔健康特征值为{value}，建议：父母口腔健康状况直接影响孩子，建议父母定期进行口腔健康检查。")
                    elif feature == 'parent_emphasis':
                        specific_suggestions.append(f"父母重视程度特征值为{value}，建议：父母应高度重视孩子的口腔健康，将其纳入日常护理的一部分。鼓励孩子养成良好的口腔卫生习惯。")
                    elif feature == 'whether_cure':
                        specific_suggestions.append(f"是否治疗特征值为{value}，建议：如果发现龋齿或其他问题，应尽快治疗，避免延误导致更严重的后果。")
                    elif feature == 'age':
                        specific_suggestions.append(f"年龄为{value}岁，建议：根据年龄调整护理方式：0-2岁：用湿纱布擦拭牙面。3-5岁：开始学习刷牙，家长协助。6-12岁：培养自主刷牙，定期检查。")
                    elif feature == 'height':
                        specific_suggestions.append(f"身高为{value}cm，建议：无")
                    elif feature == 'weight':
                        specific_suggestions.append(f"体重为{value}kg，建议：无")
                    elif feature == 'brush_time':
                        specific_suggestions.append(f"刷牙时间为{value}分钟，建议：每次刷牙至少2分钟，确保每个牙面都被清洁到。可使用计时器或儿童刷牙APP帮助孩子掌握时间。")
                    elif feature == 'brush_counter':
                        specific_suggestions.append(f"每天刷牙次数为{value}次，建议：每天至少刷牙两次（早晚各一次），餐后可用清水或儿童漱口水漱口。")
                    elif feature == 'milk_time':
                        specific_suggestions.append(f"晚上刷完牙喝奶的时间为{value}，建议：晚上刷牙后避免再进食或喝奶，防止奶瓶龋。")


                # 生成解释文本
                explanation = f"根据您的输入数据，系统预测您患龋齿的概率为{prob:.2%}。\n\n"

                # 添加通用建议
                explanation += "通用建议：\n"
                for i, suggestion in enumerate(general_suggestions, 1):
                    explanation += f"{i}. {suggestion}\n"

                # 添加特征贡献和针对性建议
                explanation += "\n对您风险贡献最大的三个因素是：\n"
                for i, (_, row) in enumerate(top_3_features.iterrows(), 1):
                    explanation += f"{i}. {row['特征']} (贡献度: {row['SHAP值']:.4f})\n"

                explanation += "\n针对性建议：\n"
                for i, suggestion in enumerate(specific_suggestions, 1):
                    explanation += f"{i}. {suggestion}\n"

            except Exception as e:
                print(f"计算SHAP值时出错: {str(e)}")
                explanation = self.generate_explanation(input_df, prob)

            return {
                'probability': prob,
                'risk_level': risk,
                'explanation': explanation
            }

        except Exception as e:
            print(f"预测过程中出错: {str(e)}")
            # 返回默认的低风险结果
            return {
                'probability': 0.3,
                'risk_level': "低风险",
                'explanation': "由于数据处理过程中出现错误，系统返回默认的低风险结果。请检查输入数据的格式是否正确。"
            }

    def visualize_tree(self, tree_index=0):
        """
        可视化XGBoost中的单棵树

        参数：
        tree_index: 要可视化的树的索引，默认为0（第一棵树）
        """
        try:
            # 检查是否安装了graphviz
            try:
                import graphviz
            except ImportError:
                print("\n错误：未安装graphviz包，无法可视化决策树")
                print("请按以下步骤安装：")
                print("1. 安装graphviz软件：")
                print("   - Windows: 从 https://graphviz.org/download/ 下载并安装")
                print("   - 确保将graphviz的bin目录添加到系统PATH环境变量")
                print("2. 安装Python包：")
                print("   pip install graphviz")
                return

            # 获取基础模型
            base_model = self.model

            print("\n=== 模型结构分析 ===")
            print(f"模型类型: {type(base_model)}")

            # 获取树的数量
            n_trees = len(base_model.get_booster().get_dump())
            print(f"\n模型中共有 {n_trees} 棵树")

            # 检查树索引是否有效
            if tree_index >= n_trees:
                print(f"错误：树索引 {tree_index} 超出范围（最大索引：{n_trees-1}）")
                return

            # 绘制指定的树
            print(f"\n正在绘制第 {tree_index + 1} 棵树...")
            xgb.plot_tree(base_model, num_trees=tree_index)

            # 显示树的结构
            print("\n=== 树的结构 ===")
            print(base_model.get_booster().get_dump()[tree_index])

        except Exception as e:
            print(f"\n可视化树结构时出错: {str(e)}")
            print("\n请确保：")
            print("1. 已正确安装graphviz软件")
            print("2. graphviz的bin目录已添加到系统PATH")
            print("3. 已安装Python的graphviz包")
            print("\n如果问题仍然存在，可以尝试：")
            print("1. 重新安装graphviz")
            print("2. 重启Python环境")
            print("3. 检查系统环境变量设置")

    def analyze_feature_importance(self):
        """
        分析并显示模型中最重要的特征
        """
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

            # 获取特征重要性
            importance = self.model.feature_importances_

            # 创建特征重要性DataFrame
            feature_importance = pd.DataFrame({
                '特征': self.features,
                '重要性': importance
            })

            # 按重要性排序
            feature_importance = feature_importance.sort_values('重要性', ascending=False)

            # 计算重要性百分比
            total_importance = feature_importance['重要性'].sum()
            feature_importance['重要性百分比'] = (feature_importance['重要性'] / total_importance * 100).round(2)

            # 显示所有特征的重要性
            print("\n=== 所有特征的重要性排序 ===")
            print(feature_importance.to_string(index=False))

            # 绘制特征重要性条形图
            plt.figure(figsize=(15, 8))
            plt.bar(range(len(importance)), feature_importance['重要性百分比'])
            plt.xticks(range(len(importance)), feature_importance['特征'], rotation=45, ha='right')
            plt.title('所有特征的贡献百分比')
            plt.ylabel('重要性百分比 (%)')
            plt.tight_layout()
            plt.show()

            return feature_importance

        except Exception as e:
            print(f"\n分析特征重要性时出错: {str(e)}")
            print("\n请确保：")
            print("1. 已安装中文字体（如SimHei）")
            print("2. matplotlib配置正确")
            print("\n如果问题仍然存在，可以尝试：")
            print("1. 安装中文字体")
            print("2. 修改matplotlib的字体设置")
            print("3. 重启Python环境")
            return None

    def analyze_feature_correlation(self):
        """
        分析特征与目标变量的相关性
        """
        if not hasattr(self, 'training_data') or self.training_data is None:
            print("错误：没有可用的训练数据，请先训练模型")
            return None

        # 计算特征与目标变量的相关性
        correlation = pd.DataFrame({
            '特征': self.features,
            '相关性': [self.training_data[feature].corr(self.target_variable)
                      for feature in self.features]
        })

        # 按相关性绝对值排序
        correlation['相关性绝对值'] = correlation['相关性'].abs()
        correlation = correlation.sort_values('相关性绝对值', ascending=False)

        print("\n=== 特征相关性分析 ===")
        print("特征与目标变量的相关性（按绝对值排序）:")
        print(correlation.to_string(index=False))

        return correlation

    def remove_low_importance_features(self, n_features_to_remove=10):
        """
        将贡献度最低的特征权重设为0并重新训练模型

        参数：
        n_features_to_remove: 要处理的特征数量，默认为10
        """
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            # 分析特征重要性
            feature_importance = self.analyze_feature_importance()
            if feature_importance is None:
                return False

            # 获取贡献度最低的特征
            low_importance_features = feature_importance.tail(n_features_to_remove)

            print(f"\n=== 将贡献度最低的 {n_features_to_remove} 个特征的权重设为0 ===")
            print("将被处理的特征:")
            for i, (_, row) in enumerate(low_importance_features.iterrows(), 1):
                print(f"{i}. {row['特征']} (重要性: {row['重要性百分比']:.2f}%)")

            # 确认是否继续
            confirm = input("\n是否确认将这些特征的权重设为0并重新训练模型？(y/n): ").lower()
            if confirm != 'y':
                print("操作已取消")
                return False

            # 获取要处理的特征名称列表
            features_to_zero = low_importance_features['特征'].tolist()

            # 创建特征权重字典
            feature_weights = {feature: 1.0 for feature in self.features}
            for feature in features_to_zero:
                feature_weights[feature] = 0.0

            print("\n开始使用调整后的特征权重重新训练模型...")

            # 划分训练集、验证集和测试集
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.training_data[self.features],
                self.target_variable,
                test_size=0.3,
                random_state=42,
                stratify=self.target_variable
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=0.5,
                random_state=42,
                stratify=y_temp
            )

            # 应用特征权重
            X_train_weighted = X_train.copy()
            X_val_weighted = X_val.copy()
            X_test_weighted = X_test.copy()

            for feature in self.features:
                weight = feature_weights[feature]
                X_train_weighted[feature] = X_train[feature] * weight
                X_val_weighted[feature] = X_val[feature] * weight
                X_test_weighted[feature] = X_test[feature] * weight

            # 训练模型
            self.train_model(X_train_weighted, y_train)

            # 评估模型
            print("\n=== 模型评估结果 ===")
            print("\n训练集评估:")
            self.evaluate_model(X_train_weighted, y_train, "训练集")

            print("\n验证集评估:")
            self.evaluate_model(X_val_weighted, y_val, "验证集")

            print("\n测试集评估:")
            self.evaluate_model(X_test_weighted, y_test, "测试集")

            # 保存新模型
            self.save_model(MODEL_FILE)
            print(f"\n新模型已保存到 {MODEL_FILE}")

            return True

        except Exception as e:
            print(f"\n处理低贡献度特征时出错: {str(e)}")
            return False

    def adjust_feature_weights(self):
        """
        调整特征权重
        """
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            # 分析特征重要性
            feature_importance = self.analyze_feature_importance()
            if feature_importance is None:
                return False

            # 获取前10个和后10个特征
            top_10_features = feature_importance.head(10)
            bottom_10_features = feature_importance.tail(10)

            print("\n=== 特征权重调整 ===")
            print("前10个重要特征:")
            for i, (_, row) in enumerate(top_10_features.iterrows(), 1):
                print(f"{i}. {row['特征']} (重要性: {row['重要性百分比']:.2f}%)")

            print("\n后10个次要特征:")
            for i, (_, row) in enumerate(bottom_10_features.iterrows(), 1):
                print(f"{i}. {row['特征']} (重要性: {row['重要性百分比']:.2f}%)")

            # 确认是否继续
            confirm = input("\n是否确认调整这些特征的权重？(y/n): ").lower()
            if confirm != 'y':
                print("操作已取消")
                return False

            # 调整特征权重
            X_train = self.training_data.copy()

            # 增加前10个特征的权重
            for feature in top_10_features['特征']:
                X_train[feature] = X_train[feature] * 1.5

            # 减少后10个特征的权重
            for feature in bottom_10_features['特征']:
                X_train[feature] = X_train[feature] * 0.5

            # 划分训练集、验证集和测试集
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_train,
                self.target_variable,
                test_size=0.3,
                random_state=42,
                stratify=self.target_variable
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=0.5,
                random_state=42,
                stratify=y_temp
            )

            # 重新训练模型
            print("\n开始使用调整后的特征权重重新训练模型...")
            self.train_model(X_train, y_train)

            # 评估模型
            print("\n=== 模型评估结果 ===")
            print("\n训练集评估:")
            self.evaluate_model(X_train, y_train, "训练集")

            print("\n验证集评估:")
            self.evaluate_model(X_val, y_val, "验证集")

            print("\n测试集评估:")
            self.evaluate_model(X_test, y_test, "测试集")

            # 保存新模型
            self.save_model(MODEL_FILE)
            print(f"\n新模型已保存到 {MODEL_FILE}")

            return True

        except Exception as e:
            print(f"\n调整特征权重时出错: {str(e)}")
            return False

    def filter_input_features(self, input_data):
        """
        过滤用户输入的特征，只保留当前模型需要的特征

        参数：
        input_data: 包含所有用户输入特征的字典

        返回：
        dict: 只包含当前模型需要的特征的字典
        """
        filtered_data = {}

        # 1. 处理分类变量
        for feature in self.features:
            if feature in self.label_encoders:
                # 如果是分类变量
                if feature in input_data:
                    try:
                        # 尝试转换分类变量
                        value = input_data[feature]
                        if value in self.label_encoders[feature].classes_:
                            # 将分类变量转换为数值
                            filtered_data[feature] = float(self.label_encoders[feature].transform([value])[0])
                        else:
                            print(f"警告：特征 {feature} 的值 {value} 不在有效范围内，使用默认值")
                            filtered_data[feature] = float(self.label_encoders[feature].transform([self.label_encoders[feature].classes_[0]])[0])
                    except Exception as e:
                        print(f"警告：无法处理分类变量 {feature}，使用默认值")
                        filtered_data[feature] = float(self.label_encoders[feature].transform([self.label_encoders[feature].classes_[0]])[0])
                else:
                    # 如果特征缺失，使用默认值
                    print(f"注意：特征 {feature} 未输入，使用默认值")
                    filtered_data[feature] = float(self.label_encoders[feature].transform([self.label_encoders[feature].classes_[0]])[0])
            else:
                # 如果是连续变量
                if feature in input_data:
                    try:
                        # 尝试转换为浮点数
                        filtered_data[feature] = float(input_data[feature])
                    except (ValueError, TypeError):
                        print(f"警告：特征 {feature} 的值无法转换为数值，使用默认值0")
                        filtered_data[feature] = 0.0
                else:
                    # 如果特征缺失，使用默认值0
                    print(f"注意：特征 {feature} 未输入，使用默认值0")
                    filtered_data[feature] = 0.0

        # 2. 创建交互特征（如果需要的特征存在）
        if all(f in filtered_data for f in ['brush_method', 'floss_seq', 'wash_seq']):
            filtered_data['oral_hygiene_score'] = (
                filtered_data['brush_method'] +
                filtered_data['floss_seq'] +
                filtered_data['wash_seq']
            )

        if 'sweet_seq' in filtered_data and 'oral_hygiene_score' in filtered_data:
            filtered_data['sweet_hygiene_interaction'] = (
                filtered_data['sweet_seq'] *
                filtered_data['oral_hygiene_score']
            )

        if 'parent_edu' in filtered_data and 'oral_hygiene_score' in filtered_data:
            filtered_data['parent_edu_hygiene'] = (
                filtered_data['parent_edu'] *
                filtered_data['oral_hygiene_score']
            )

        if all(f in filtered_data for f in ['first_check', 'first_decsyed']):
            filtered_data['check_to_decsyed_interval'] = (
                filtered_data['first_decsyed'] -
                filtered_data['first_check']
            )

        return filtered_data

    def single_prediction(self):
        """
        单次预测流程

        功能：
        1. 检查模型是否已加载
        2. 收集用户输入的特征值（包括所有原始特征）
        3. 处理缺失值（使用默认值）
        4. 进行预测并显示结果
        5. 显示预测解释
        """
        print("\n===== 单次预测 =====")

        # 检查模型是否已加载
        if self.model is None or self.imputer is None:
            print("错误：模型未加载，请先训练或加载模型")
            return

        # 确保original_features已初始化
        if not hasattr(self, 'original_features') or not self.original_features:
            # 如果original_features未初始化，则使用features
            self.original_features = self.features.copy()
            print("注意：使用当前模型特征作为原始特征")

        input_data = {}  # 存储用户输入

        # 获取所有原始特征的用户输入
        all_features = self.original_features  # 获取所有原始特征

        # 获取用户输入
        for feature in all_features:
            while True:
                value = input(f"{feature} (留空使用默认值): ").strip()

                if value == "":
                    # 使用默认值(分类变量使用最常见的类别，数值变量使用0)
                    if feature in self.label_encoders:
                        default = self.label_encoders[feature].classes_[0]
                    else:
                        default = 0
                    input_data[feature] = default
                    break

                try:
                    # 处理分类变量
                    if feature in self.label_encoders:
                        if value in self.label_encoders[feature].classes_:
                            input_data[feature] = value  # 保存原始值
                        else:
                            print(f"无效值，可选: {list(self.label_encoders[feature].classes_)}")
                            continue
                    else:
                        # 处理数值变量
                        input_data[feature] = float(value)
                    break
                except ValueError:
                    print("请输入有效值")
                    continue

        # 过滤输入特征，只保留当前模型需要的特征
        filtered_data = self.filter_input_features(input_data)

        # 进行预测
        try:
            result = self.predict_risk(filtered_data)
            if result:
                print("\n预测结果：")
                print(f"风险概率: {result['probability']:.2%}")
                print(f"风险等级: {result['risk_level']}")

                print("\n针对性建议：")
                for suggestion in result['explanation'].split('\n'):
                    print(f"- {suggestion}")
        except Exception as e:
            print(f"预测过程中出错: {str(e)}")


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
    3. 可视化第一棵树
    4. 分析特征重要性
    5. 分析树结构
    6. 移除低贡献度特征
    7. 调整特征权重
    8. 退出：结束程序
    """
    print("===== 口腔健康风险预测系统 =====")
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
        print("3. 可视化第一棵树")
        print("4. 分析特征重要性")
        print("5. 分析树结构")
        print("6. 移除低贡献度特征")
        print("7. 调整特征权重")
        print("8. 退出")

        choice = input("请选择操作: ").strip()

        if choice == '1':
            predictor.single_prediction()  # 单次预测
        elif choice == '2':
            batch_prediction(predictor)  # 批量预测
        elif choice == '3':
            predictor.visualize_tree(0)  # 可视化第一棵树
        elif choice == '4':
            predictor.analyze_feature_importance()  # 分析特征重要性
        elif choice == '5':
            predictor.visualize_tree()  # 分析树结构
        elif choice == '6':
            predictor.remove_low_importance_features()  # 移除低贡献度特征
        elif choice == '7':
            predictor.adjust_feature_weights()  # 调整特征权重
        elif choice == '8':
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
        # 首先划分训练集和临时测试集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        # 将临时测试集进一步划分为验证集和测试集
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        print("\n数据集划分情况:")
        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        print(f"测试集大小: {len(X_test)}")

        # 4. 训练模型
        predictor.model = predictor.train_model(X_train, y_train)

        # 保存训练数据
        predictor.training_data = X_train.copy()
        predictor.target_variable = y_train.copy()

        # 5. 评估模型
        print("\n开始模型评估与调整...")
        predictor.evaluate_and_adjust(X_train, y_train, X_val, y_val)

        print("\n训练集评估结果:")
        predictor.evaluate_model(X_train, y_train, "训练集")

        print("\n验证集评估结果:")
        predictor.evaluate_model(X_val, y_val, "验证集")

        print("\n测试集评估结果:")
        predictor.evaluate_model(X_test, y_test, "测试集")

        # 6. 保存模型
        predictor.save_model(MODEL_FILE)

    except Exception as e:
        print(f"训练过程中出错: {str(e)}")


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