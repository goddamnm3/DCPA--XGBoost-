import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from model.predictor import DentalHealthPredictor
from utils.data_processor import get_feature_statistics

class DentalHealthApp:
    def __init__(self, predictor):
        """
        初始化GUI应用
        
        参数：
        predictor: 预测器实例
        """
        self.predictor = predictor
        self.root = tk.Tk()
        self.root.title("口腔健康风险预测系统")
        self.root.geometry("800x600")
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建输入区域
        self.create_input_area()
        
        # 创建按钮区域
        self.create_button_area()
        
        # 创建结果显示区域
        self.create_result_area()
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

    def create_input_area(self):
        """创建输入区域"""
        input_frame = ttk.LabelFrame(self.main_frame, text="输入特征", padding="5")
        input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 创建滚动区域
        canvas = tk.Canvas(input_frame)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 创建输入字段
        self.input_vars = {}
        row = 0
        
        # 分类变量
        categorical_features = [
            "gender", "brush_method", "toothpaste", "wash_meal",
            "floss_seq", "wash_seq", "sweet_seq", "sweet_drink_seq",
            "other_snack_seq", "dental_seq", "first_brush", "help_until",
            "first_check", "first_descsyed", "descsyed_cure", "mike_method",
            "whether_F", "sealant", "systemic_disease", "self_brush",
            "parent_edu", "parent_job", "parent_health", "parent_emphasis",
            "whether_cure"
        ]
        
        for feature in categorical_features:
            if feature in self.predictor.label_encoders:
                ttk.Label(scrollable_frame, text=feature).grid(row=row, column=0, sticky=tk.W, pady=2)
                var = tk.StringVar()
                self.input_vars[feature] = var
                combo = ttk.Combobox(scrollable_frame, textvariable=var, state="readonly")
                combo['values'] = list(self.predictor.label_encoders[feature].classes_)
                combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
                row += 1
        
        # 连续变量
        continuous_features = [col for col in self.predictor.features 
                             if col not in categorical_features]
        
        for feature in continuous_features:
            ttk.Label(scrollable_frame, text=feature).grid(row=row, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar()
            self.input_vars[feature] = var
            entry = ttk.Entry(scrollable_frame, textvariable=var)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1
        
        # 配置滚动区域
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 配置网格权重
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)
        scrollable_frame.columnconfigure(1, weight=1)

    def create_button_area(self):
        """创建按钮区域"""
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(button_frame, text="预测", command=self.predict).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清除", command=self.clear_inputs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

    def create_result_area(self):
        """创建结果显示区域"""
        result_frame = ttk.LabelFrame(self.main_frame, text="预测结果", padding="5")
        result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 创建文本区域
        self.result_text = tk.Text(result_frame, wrap=tk.WORD, width=70, height=15)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        # 配置网格权重
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

    def predict(self):
        """执行预测"""
        try:
            # 收集输入数据
            input_data = {}
            for feature, var in self.input_vars.items():
                value = var.get()
                if value:
                    input_data[feature] = value
            
            # 检查是否有输入
            if not input_data:
                messagebox.showwarning("警告", "请至少输入一个特征值")
                return
            
            # 进行预测
            result = self.predictor.predict_risk(input_data)
            
            # 显示结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"预测概率: {result['probability']:.2%}\n")
            self.result_text.insert(tk.END, f"风险等级: {result['risk_level']}\n\n")
            self.result_text.insert(tk.END, "个性化建议:\n")
            self.result_text.insert(tk.END, result['explanation'])
            
        except Exception as e:
            messagebox.showerror("错误", f"预测过程中出错: {str(e)}")

    def clear_inputs(self):
        """清除所有输入"""
        for var in self.input_vars.values():
            var.set("")
        self.result_text.delete(1.0, tk.END)

    def run(self):
        """运行应用程序"""
        self.root.mainloop() 