import pandas as pd
import numpy as np
import re

def preprocess_user_input(input_data):
    """
    处理用户输入的数据，使其符合模型要求

    参数:
        input_data: 字典，包含用户输入的特征值

    返回:
        处理后的数据字典
    """
    processed_data = {}
    
    # 1. 处理数值列
    numeric_cols = ['age', 'height', 'weight', 'brush_time', 'brush_counter',
                    'descsyed_counter', 'milk_time']
    
    def extract_numeric_value(x):
        """提取数值或计算区间中值"""
        if pd.isna(x) or x == '':
            return np.nan
            
        # 转换为字符串处理
        x = str(x)
        
        # 处理中文单位
        x = re.sub(r'[岁厘米cm千克kg]', '', x)
        
        # 处理区间值
        seps = ['～', '~', '-']
        sep = next((s for s in seps if s in x), None)
        if sep:
            parts = x.split(sep)
            num1 = float(re.sub(r'[^\d.]', '', parts[0]).strip() or 0)
            num2 = float(re.sub(r'[^\d.]', '', parts[1]).strip() or 0)
            return (num1 + num2) / 2
            
        # 处理单个数值
        else:
            cleaned = re.sub(r'[^\d.]', '', x)
            return float(cleaned.strip()) if cleaned.strip() else np.nan
    
    # 处理数值列
    for col in numeric_cols:
        if col in input_data:
            processed_data[col] = extract_numeric_value(input_data[col])
    
    # 2. 特殊处理descsyed_counter列
    if 'descsyed_counter' in input_data:
        def transform_descsyed_counter(x):
            if pd.isna(x) or x == '':
                return np.nan
            x = str(x)
            
            # 转换选项为数值
            if x.startswith('A.'):
                return 0
            elif x.startswith('B.'):
                return 1.5
            elif x.startswith('C.'):
                return 3.5
            elif x.startswith('D.'):
                return 5.5
            elif x.startswith('E.'):
                return 7.5
            elif x.startswith('F.'):
                return 9.5
            elif x.startswith('G.'):
                return 11.5
            elif x.startswith('H.'):
                return 13.5
            elif x.startswith('I.'):
                return 15.5
            elif x.startswith('J.'):
                return 17.5
            elif x.startswith('K.'):
                return 19
            else:
                return extract_numeric_value(x)
        
        processed_data['descsyed_counter'] = transform_descsyed_counter(input_data['descsyed_counter'])
    
    # 3. 处理分类列
    # 分类变量的选项映射（从自定义选项到数值编码）
    categorical_mappings = {
        'gender': {'男': 0, '女': 1},
        'brush_method': {'横刷法': 0, '竖刷法': 1, '旋转法': 2, '其他方法': 3},
        'toothpaste': {'含氟牙膏': 0, '不含氟牙膏': 1, '其他牙膏': 2},
        'wash_meal': {'餐后立即': 0, '餐后延迟': 1, '不固定': 2},
        'floss_seq': {'每天使用': 0, '偶尔使用': 1, '从不使用': 2},
        'wash_seq': {'每天使用': 0, '偶尔使用': 1, '从不使用': 2},
        'sweet_seq': {'每天食用': 0, '每周几次': 1, '很少食用': 2, '从不食用': 3},
        'sweet_drink_seq': {'每天饮用': 0, '每周几次': 1, '很少饮用': 2, '从不饮用': 3},
        'other_snack_seq': {'每天食用': 0, '每周几次': 1, '很少食用': 2, '从不食用': 3},
        'dental_seq': {'每年检查': 0, '每两年检查': 1, '很少检查': 2, '从不检查': 3},
        'first_brush': {'1岁前': 0, '1-2岁': 1, '2-3岁': 2, '3岁后': 3},
        'help_until': {'3岁前': 0, '3-6岁': 1, '6-9岁': 2, '9岁后': 3},
        'first_check': {'1岁前': 0, '1-2岁': 1, '2-3岁': 2, '3岁后': 3},
        'first_decsyed': {'1岁前': 0, '1-2岁': 1, '2-3岁': 2, '3岁后': 3},
        'descsyed_cure': {'及时治疗': 0, '延迟治疗': 1, '未治疗': 2},
        'mike_method': {'奶瓶': 0, '杯子': 1, '其他方式': 2},
        'whether_F': {'是': 0, '否': 1},
        'sealant': {'已做': 0, '未做': 1},
        'systemic_disease': {'有': 0, '无': 1},
        'self_brush': {'是': 0, '否': 1},
        'parent_edu': {'小学': 0, '初中': 1, '高中': 2, '大学及以上': 3},
        'parent_job': {'工人': 0, '农民': 1, '教师': 2, '其他': 3},
        'family_income': {'低': 0, '中': 1, '高': 2},
        'parent_health': {'好': 0, '一般': 1, '差': 2},
        'parent_emphasis': {'非常重视': 0, '一般重视': 1, '不太重视': 2},
        'whether_cure': {'是': 0, '否': 1},
        'F_seq': {'每天使用': 0, '偶尔使用': 1, '从不使用': 2}
    }
    
    categorical_cols = list(categorical_mappings.keys())
    
    for col in categorical_cols:
        if col in input_data:
            value = str(input_data[col])
            # 使用映射转换选项为数值
            if value in categorical_mappings[col]:
                processed_data[col] = categorical_mappings[col][value]
            else:
                # 如果没有匹配的选项，尝试直接转换为数值
                try:
                    processed_data[col] = float(value)
                except:
                    processed_data[col] = 0
    
    # 4. 处理缺失值
    # 对于height和weight，使用默认值
    if 'height' not in processed_data or pd.isna(processed_data['height']):
        processed_data['height'] = 150.0  # 默认身高
    if 'weight' not in processed_data or pd.isna(processed_data['weight']):
        processed_data['weight'] = 50.0   # 默认体重
    
    # 其他数值列用0填充
    for col in numeric_cols:
        if col not in processed_data or pd.isna(processed_data[col]):
            processed_data[col] = 0.0
    
    # 分类列用0填充
    for col in categorical_cols:
        if col not in processed_data or pd.isna(processed_data[col]):
            processed_data[col] = 0
    
    # 5. 增加if_illness列
    if 'descsyed_counter' in processed_data:
        processed_data['if_illness'] = 1 if processed_data['descsyed_counter'] > 0 else 0
    else:
        processed_data['if_illness'] = 0
    
    return processed_data 