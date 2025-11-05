import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def load_data(file_path):
    """
    加载数据文件
    
    参数：
    file_path: 数据文件路径
    
    返回：
    DataFrame: 处理后的数据
    """
    try:
        data = pd.read_excel(file_path)
        print("数据已成功加载！样本数:", len(data))
        print("\n目标变量分布情况:")
        print(data['if_illness'].value_counts(normalize=True))
        
        # 删除不必要的列
        cols_to_drop = ['writer_id', 'descsyed_counter']
        data = data.drop([col for col in cols_to_drop if col in data.columns], axis=1)
        
        # 确保目标变量为数值型
        data['if_illness'] = pd.to_numeric(data['if_illness'], errors='coerce')
        
        return data
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise

def preprocess_data(data, predictor):
    """
    数据预处理
    
    参数：
    data: 原始数据
    predictor: 预测器实例
    
    返回：
    tuple: (X, y) 处理后的特征矩阵和目标变量
    """
    try:
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

        print("\n正在编码分类变量...")
        for col in categorical_features:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                predictor.label_encoders[col] = le

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

        predictor.features = [col for col in data.columns if col != 'if_illness']
        X = data[predictor.features]
        y = data["if_illness"]

        print("正在处理缺失值...")
        predictor.imputer = SimpleImputer(strategy='most_frequent')
        X = pd.DataFrame(predictor.imputer.fit_transform(X), columns=X.columns)

        predictor.original_features = predictor.features.copy()
        return X, y
        
    except Exception as e:
        print(f"数据预处理过程中出错: {str(e)}")
        raise

def create_interaction_features(data):
    """
    创建交互特征
    
    参数：
    data: 原始数据
    
    返回：
    DataFrame: 添加了交互特征的数据
    """
    try:
        # 创建口腔卫生评分
        if all(col in data.columns for col in ['brush_method', 'floss_seq', 'wash_seq']):
            data['oral_hygiene_score'] = (
                data['brush_method'].astype(float) + 
                data['floss_seq'].astype(float) + 
                data['wash_seq'].astype(float)
            )
        
        # 创建甜食与口腔卫生的交互特征
        if 'sweet_seq' in data.columns and 'oral_hygiene_score' in data.columns:
            data['sweet_hygiene_interaction'] = (
                data['sweet_seq'].astype(float) * 
                data['oral_hygiene_score'].astype(float)
            )
        
        # 创建父母教育与口腔卫生的交互特征
        if 'parent_edu' in data.columns and 'oral_hygiene_score' in data.columns:
            data['parent_edu_hygiene'] = (
                data['parent_edu'].astype(float) * 
                data['oral_hygiene_score'].astype(float)
            )
        
        # 创建首次检查与首次龋齿的时间间隔
        if all(col in data.columns for col in ['first_check', 'first_descsyed']):
            data['check_to_decsyed_interval'] = (
                data['first_descsyed'].astype(float) - 
                data['first_check'].astype(float)
            )
            
        return data
        
    except Exception as e:
        print(f"创建交互特征时出错: {str(e)}")
        return data

def handle_missing_values(data, strategy='most_frequent'):
    """
    处理缺失值
    
    参数：
    data: 原始数据
    strategy: 填充策略，可选 'mean', 'median', 'most_frequent'
    
    返回：
    DataFrame: 处理后的数据
    """
    try:
        imputer = SimpleImputer(strategy=strategy)
        return pd.DataFrame(
            imputer.fit_transform(data),
            columns=data.columns
        )
    except Exception as e:
        print(f"处理缺失值时出错: {str(e)}")
        return data

def get_feature_statistics(data):
    """
    获取特征统计信息
    
    参数：
    data: 原始数据
    
    返回：
    dict: 包含每个特征的统计信息
    """
    try:
        stats = {}
        for col in data.columns:
            if col != 'if_illness':
                if data[col].dtype in ['object', 'category']:
                    # 对于分类变量，获取最常见的类别
                    stats[col] = {
                        'type': 'categorical',
                        'most_common': data[col].mode().iloc[0]
                    }
                else:
                    # 对于连续变量，获取中位数
                    stats[col] = {
                        'type': 'continuous',
                        'median': data[col].median()
                    }
        return stats
    except Exception as e:
        print(f"获取特征统计信息时出错: {str(e)}")
        return {} 