import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import xgboost as xgb
import joblib
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(file_path):
    """
    加载数据

    参数：
    file_path: 数据文件路径

    返回：
    DataFrame: 加载的数据
    """
    print("正在加载数据...")
    data = pd.read_excel(file_path)
    print(f"数据加载完成，共 {len(data)} 条记录")

    # 检查目标变量分布
    print("\n目标变量分布：")
    print(data['if_illness'].value_counts(normalize=True))

    # 移除不必要的列
    cols_to_drop = ['writer_id', 'descsyed_counter']
    data = data.drop([col for col in cols_to_drop if col in data.columns], axis=1)

    # 确保目标变量是数值型
    data['if_illness'] = pd.to_numeric(data['if_illness'], errors='coerce')

    return data


def preprocess_data(data, predictor):
    """
    预处理数据

    参数：
    data: 原始数据
    predictor: 预测器实例

    返回：
    tuple: (X, y) 处理后的特征矩阵和目标变量
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

    # 1. 分类变量编码
    print("\n正在编码分类变量...")
    for col in categorical_features:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            predictor.label_encoders[col] = le

    # 2. 连续变量分箱
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

    # 3. 特征选择
    predictor.features = [col for col in data.columns if col != 'if_illness']
    X = data[predictor.features]
    y = data["if_illness"]

    # 4. 处理缺失值
    print("正在处理缺失值...")
    predictor.imputer = SimpleImputer(strategy='most_frequent')
    X = pd.DataFrame(predictor.imputer.fit_transform(X), columns=X.columns)

    predictor.original_features = predictor.features.copy()
    return X, y


def train_model(X, y):
    """
    训练模型

    参数：
    X: 特征矩阵
    y: 目标变量

    返回：
    XGBoost模型
    """
    try:
        # 1. 使用贝叶斯优化搜索最佳参数
        print("\n开始贝叶斯优化参数搜索...")

        # 定义参数空间
        param_space = {
            'learning_rate': Real(0.01, 0.3),
            'max_depth': Integer(3, 10),
            'n_estimators': Integer(100, 1000),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'min_child_weight': Integer(1, 7),
            'gamma': Real(0, 1.0),
            'reg_alpha': Real(0, 1.0),
            'reg_lambda': Real(0, 1.0)
        }

        # 创建基础模型
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )

        # 使用贝叶斯优化
        bayes_search = BayesSearchCV(
            estimator=base_model,
            search_spaces=param_space,
            n_iter=50,
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
            **best_params
        }

        print("\n开始训练最终模型...")
        model = xgb.XGBClassifier(**final_params)
        model.fit(X, y)
        print("模型训练完成")

        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

        return model

    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        # 尝试使用更简单的参数重新训练
        try:
            print("\n尝试使用更简单的参数重新训练...")
            simple_params = {
                'objective': 'binary:logistic',
                'max_depth': 5,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.3,
                'reg_alpha': 0.2,
                'reg_lambda': 1.0,
                'random_state': 42,
                'tree_method': 'hist'
            }
            model = xgb.XGBClassifier(**simple_params)
            model.fit(X, y)
            print("使用简单参数训练成功")
            return model
        except Exception as e2:
            print(f"使用简单参数训练也失败: {str(e2)}")
            raise


def train_new_model(predictor, data_file):
    """
    训练新模型

    参数：
    predictor: 预测器实例
    data_file: 数据文件路径
    """
    try:
        # 1. 加载数据
        data = load_data(data_file)

        # 2. 预处理
        X, y = preprocess_data(data, predictor)

        # 3. 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        print("\n数据集划分情况:")
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")

        # 4. 训练模型
        predictor.model = train_model(X_train, y_train)
        predictor.training_data = X_train.copy()
        predictor.target_variable = y_train.copy()

        # 5. 评估模型
        print("\n开始模型评估...")
        evaluate_model(predictor.model, X_train, y_train, "训练集")
        evaluate_model(predictor.model, X_test, y_test, "测试集")

        # 6. 保存模型
        predictor.save_model("dental_model.pkl")

        return predictor

    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        raise


def evaluate_model(model, X, y, dataset_name):
    """
    评估模型性能

    参数：
    model: 训练好的模型
    X: 特征矩阵
    y: 目标变量
    dataset_name: 数据集名称
    """
    # 获取预测结果
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y, y_pred)
    auc_roc = roc_auc_score(y, y_pred_proba)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    # 打印评估结果
    print(f"\n{dataset_name} 模型评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")

    # 打印特征重要性
    if hasattr(model, 'feature_importances_'):
        print("\n特征重要性(前20个):")
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        for i in range(min(20, len(X.columns))):
            print(f"{X.columns[indices[i]]}: {importance[indices[i]]:.4f}")

# 移除所有绘图相关的函数