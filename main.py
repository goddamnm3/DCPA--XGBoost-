import os
import sys
import pandas as pd

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from model.predictor import DentalHealthPredictor
from model.trainer import train_new_model


def main():
    """
    主程序入口
    """
    print("===== 口腔健康风险预测系统 =====")
    predictor = DentalHealthPredictor()  # 创建预测器实例
    fixed_data_path = "D:\\dachuang\\processed_data2.xlsx"  # 固定的训练数据路径

    # 检查是否已有训练好的模型
    if os.path.exists("dental_model.pkl"):
        choice = input("检测到已有模型，是否重新训练？(y/n): ").lower()
        if choice == 'y':
            if not os.path.exists(fixed_data_path):
                print(f"错误：训练数据文件 {fixed_data_path} 不存在")
                return
            train_new_model(predictor, fixed_data_path)  # 训练新模型
        else:
            try:
                predictor.load_model("dental_model.pkl")  # 加载已有模型
            except Exception as e:
                print(f"加载模型时出错: {str(e)}")
                print("将重新训练模型...")
                if not os.path.exists(fixed_data_path):
                    print(f"错误：训练数据文件 {fixed_data_path} 不存在")
                    return
                train_new_model(predictor, fixed_data_path)
    else:
        print("未检测到已有模型，将进行训练...")
        if not os.path.exists(fixed_data_path):
            print(f"错误：训练数据文件 {fixed_data_path} 不存在")
            return
        train_new_model(predictor, fixed_data_path)  # 训练新模型

    # 进入预测循环
    while True:
        print("\n===== 预测菜单 =====")
        print("1. 单次预测")
        print("2. 退出")

        choice = input("请选择操作: ").strip()

        if choice == '1':
            single_prediction(predictor)  # 单次预测
        elif choice == '2':
            print("退出系统...")
            break
        else:
            print("无效输入，请重新选择")


def single_prediction(predictor):
    """
    单次预测流程
    """
    print("\n===== 单次预测 =====")

    # 检查模型是否已加载
    if predictor.model is None or predictor.imputer is None:
        print("错误：模型未加载，请先训练或加载模型")
        return

    input_data = {}  # 存储用户输入

    # 获取所有特征的用户输入
    for feature in predictor.features:
        while True:
            if feature in predictor.label_encoders:
                # 分类变量
                print(f"\n{feature} 可选值: {list(predictor.label_encoders[feature].classes_)}")
                value = input(f"请输入 {feature} (留空使用默认值): ").strip()
                if value == "":
                    # 使用默认值(最常见的类别)
                    default = predictor.label_encoders[feature].classes_[0]
                    input_data[feature] = default
                    break
                if value in predictor.label_encoders[feature].classes_:
                    input_data[feature] = value
                    break
                else:
                    print("无效值，请重新输入")
            else:
                # 连续变量
                value = input(f"请输入 {feature} (留空使用默认值0): ").strip()
                if value == "":
                    input_data[feature] = 0
                    break
                try:
                    input_data[feature] = float(value)
                    break
                except ValueError:
                    print("请输入有效数字")

    # 进行预测
    try:
        result = predictor.predict_risk(input_data)
        print("\n预测结果：")
        print(f"预测概率: {result['probability']:.2%}")
        print(f"风险等级: {result['risk_level']}")
        print("\n个性化建议:")
        print(result['explanation'])
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")


if __name__ == "__main__":
    main()  # 启动主程序