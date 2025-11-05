import tkinter as tk
from tkinter import ttk, messagebox
import os
import pandas as pd
from predictionModel import DentalHealthPredictor

class RoundedEntry(ttk.Entry):
    def __init__(self, parent, width=20, **kwargs):
        super().__init__(parent, width=width, **kwargs)
        self.configure(style='Rounded.TEntry')
        
        # 创建圆角效果
        self.bind('<Configure>', self._on_configure)
        self._create_rounded_corners()
    
    def _on_configure(self, event):
        self._create_rounded_corners()
    
    def _create_rounded_corners(self):
        # 获取控件的尺寸
        width = self.winfo_width()
        height = self.winfo_height()
        
        # 设置背景色
        self.configure(background='#ffffff')

class RoundedCombobox(ttk.Combobox):
    def __init__(self, parent, width=20, **kwargs):
        super().__init__(parent, width=width, **kwargs)
        self.configure(style='Rounded.TCombobox')
        
        # 创建圆角效果
        self.bind('<Configure>', self._on_configure)
        self._create_rounded_corners()
    
    def _on_configure(self, event):
        self._create_rounded_corners()
    
    def _create_rounded_corners(self):
        # 获取控件的尺寸
        width = self.winfo_width()
        height = self.winfo_height()
        
        # 设置背景色
        self.configure(background='#ffffff')

class DentalHealthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("口腔健康风险预测系统")
        self.root.geometry("900x700")  # 主窗口尺寸设置
        
        # 颜色主题设置
        self.primary_color = "#1e88e5"  # 主蓝色
        self.secondary_color = "#bbdefb"  # 浅蓝色
        self.accent_color = "#0d47a1"  # 深蓝色
        self.bg_color = "#f5f9ff"  # 浅蓝背景色
        self.text_color = "#333333"  # 深灰色文字
        
        self.root.configure(bg=self.bg_color)  # 设置主窗口背景色
        
        # 设置应用样式
        self.style = ttk.Style()
        # 按钮样式设置
        self.style.configure("TButton", font=("微软雅黑", 12), padding=10, background=self.primary_color)
        # 标签样式设置
        self.style.configure("TLabel", font=("微软雅黑", 14), background=self.bg_color, foreground=self.text_color)
        # 标题标签样式设置
        self.style.configure("Title.TLabel", font=("微软雅黑", 28, "bold"), background=self.bg_color, foreground=self.primary_color)
        # 副标题标签样式设置
        self.style.configure("Subtitle.TLabel", font=("微软雅黑", 16), background=self.bg_color, foreground=self.accent_color)
        # 结果标签样式设置
        self.style.configure("Result.TLabel", font=("微软雅黑", 18, "bold"), background=self.bg_color, foreground=self.accent_color)
        
        # 自定义按钮样式设置
        self.style.configure("Primary.TButton", 
                            font=("微软雅黑", 12, "bold"), 
                            background=self.primary_color, 
                            foreground="#276ab3",  # 按钮文字颜色
                            padding=15)
        
        # 设置按钮悬停和点击效果
        self.style.map("Primary.TButton",
                      background=[('active', self.accent_color),  # 悬停时变为深蓝色
                                ('pressed', self.accent_color)],  # 点击时变为深蓝色
                      foreground=[('active', '#2c6fbb'),  # 悬停时文字颜色
                                ('pressed', '#448ee4')])  # 点击时文字颜色

        # 配置圆角输入框样式
        self.style.configure('Rounded.TEntry',
                           font=("微软雅黑", 11),
                           fieldbackground="#ffffff",
                           foreground=self.text_color,
                           borderwidth=0)
        
        # 配置圆角下拉框样式
        self.style.configure('Rounded.TCombobox',
                           font=("微软雅黑", 11),
                           background="#ffffff",
                           fieldbackground="#ffffff",
                           foreground=self.text_color,
                           borderwidth=0)
        
        # 设置下拉列表样式
        self.root.option_add('*TCombobox*Listbox.font', ('微软雅黑', 12))
        self.root.option_add('*TCombobox*Listbox.background', '#ffffff')
        self.root.option_add('*TCombobox*Listbox.foreground', self.text_color)
        self.root.option_add('*TCombobox*Listbox.selectBackground', self.secondary_color)
        self.root.option_add('*TCombobox*Listbox.selectForeground', self.primary_color)
        self.root.option_add('*TCombobox*Listbox.borderwidth', 0)
        
        # 初始化预测器
        self.predictor = None
        self.load_model()
        
        # 特征名称的中文映射
        self.feature_names_cn = {
            # 请根据实际情况填写特征名称的中文映射
            # 格式: '英文特征名': '中文特征名'
            'gender': '性别',
            'age': '年龄',
            'height': '身高（cm）',
            'weight': '体重（kg）',
            'brush_time': '刷牙时间（分钟）',
            'brush_method': '刷牙方法',
            'toothpaste': '刷牙时是否使用牙膏',
            'brush_counter': '每天刷牙次数',
            'wash_meal': '是否饭后漱口',
            'floss_seq': '使用牙线频率',
            'wash_seq': '漱口频率',
            'sweet_seq': '甜食摄入频率（一天几次）',
            'sweet_drink_seq': '甜饮料摄入频率（一天几次）',
            'other_snack_seq': '其他零食摄入频率（一天几次）',
            'dental_seq': '看牙医频率',
            'first_brush': '首次刷牙年龄',
            'help_until': '父母帮助刷牙到几岁',
            'first_check': '首次检查牙齿年龄',
            'first_descsyed': '首次患龋齿年龄',
            'descsyed_cure': '是否治疗龋齿',
            'milk_time': '晚上刷完牙喝奶的时间',
            'mike_method': '喂养方式',
            'whether_F': '是否进行全口涂氟',
            'F_seq': '全口涂氟频率',
            'sealant': '是否做过窝沟封闭',
            'systemic_disease': '是否患有全身性疾病',
            'self_brush': '是否自己刷牙',
            'parent_edu': '父母教育程度',
            'parent_job': '父母职业类型',
            'family_income': '家庭收入',
            'parent_health': '父母口腔健康状况',
            'parent_emphasis': '父母对口腔健康的重视程度',
            'whether_cure': '父母认为龋齿是否需要治疗',
            # 其他特征名称的中文映射
        }
        
        # 分类变量的值映射（中文选项到数值的映射）
        self.categorical_values = {
            'gender': {'男': 0, '女': 1},
            'brush_method': {'巴氏刷牙法': 0, '圆弧刷牙法': 1, '随便刷牙': 2, '其他方法': 3},
            'toothpaste': {'是': 1, '否': 0},#是否用牙膏，先对数字存疑
            'wash_meal': {'是': 1, '否': 0},
            'floss_seq': {'一天3次': 0, '一天1次': 1, '偶尔1次': 2,'不使用':3},
            'wash_seq': {'一天3次': 0, '一天1次': 1, '偶尔1次': 2,'不使用':3},
            'sweet_seq': {'0次': 0, '1次': 1, '2次': 2, '3次及以上': 3},
            'sweet_drink_seq': {'0次': 0, '1次': 1, '2次': 2, '3次及以上': 3},
            'other_snack_seq': {'0次': 0, '1次': 1, '2次': 2, '3次及以上': 3},
            'dental_seq': {'每月1次': 0, '每年1次': 1, '偶尔1次': 2, '从未去过': 3},
            'first_brush': {    # 原问题18对应的字段名
                '1岁及以下': 0,  # A
                '2岁': 1,       # B
                '3岁': 2,       # C
                '4岁': 3,       # D
                '5岁': 4,       # E
                '6岁': 5,       # F
                '7岁': 6,       # G
                '8岁': 7,       # H
                '9岁': 8,       # I
                '10岁': 9,     # J
                '11岁': 10,     # K
                '12岁': 11      # L
            },
            'help_until': {
                '1岁及以下': 0,  # A
                '2岁': 1,       # B
                '3岁': 2,       # C
                '4岁': 3,       # D
                '5岁': 4,       # E
                '6岁': 5,       # F
                '7岁': 6,       # G
                '8岁': 7,       # H
                '9岁': 8,       # I
                '10岁': 9,      # J
                '11岁': 10,     # K
                '12岁': 11      # L
            },
            'first_check': {
                '1岁及以下': 0,  # A
                '2岁': 1,       # B
                '3岁': 2,       # C
                '4岁': 3,       # D
                '5岁': 4,       # E
                '6岁': 5,       # F
                '7岁': 6,       # G
                '8岁': 7,       # H
                '9岁': 8,       # I
                '10岁': 9,      # J
                '11岁': 10,     # K
                '12岁': 11      # L
            },
            'first_descsyed': {
            '1岁及以下': 0,  # A
            '2岁': 1,       # B
            '3岁': 2,       # C
            '4岁': 3,       # D
            '5岁': 4,       # E
            '6岁': 5,       # F
            '7岁': 6,       # G
            '8岁': 7,       # H
            '9岁': 8,       # I
            '10岁': 9,      # J
            '11岁': 10,     # K
            '12岁': 11,     # L
            '没有龋齿': 12   # M
            },
            'descsyed_cure': {'是': 1, '否': 0},
            'mike_method': {'母乳喂养': 0, '奶粉喂养': 1, '混合喂养': 2},
            'whether_F': {'是': 1, '否': 0},
            'F_seq': {'没有涂过': 0, '3月/1次': 1, '6月1次': 2,'1年/1次':3},
            'sealant': {'是': 1, '否': 0},#是否做过窝沟封闭
            'systemic_disease': {'是': 1, '否': 0},#是否患有全身性疾病
            'self_brush': {'是': 1, '否': 0},#是否自己刷牙
            'parent_edu': {
                '初中及以下': 0,    # A
                '高中': 1,         # B
                '大学本科': 2,      # C
                '硕士': 3,         # D
                '博士及以上': 4     # E
            },
            'parent_job': {
                '技术类': 0,        # A
                '服务类': 1,        # B
                '医疗与健康类': 2,  # C
                '教育与培训类': 3,  # D
                '销售与市场类': 4,  # E
                '行政与管理类': 5,  # F
                '金融与会计类': 6,  # G
                '工程与创造类': 7,  # H
                '艺术与创作类': 8,  # I
                '自由职业与创业': 9  # J
            },
            'family_income': {
                '5000以下': 0,      # A
                '5000-10000': 1,    # B
                '10000-20000': 2,   # C
                '20000以上': 3,     # D
                '不固定': 4         # E
            },
            'parent_health': {
                '两人均无': 0,      # A
                '只有父亲有': 1,    # B
                '只有母亲有': 2,    # C
                '两人都有': 3       # D
            },
            'parent_emphasis': {'非常重视': 0, '一般重视': 1, '偶尔重视': 2,'不重视':3},
            'whether_cure': {'是': 1, '否': 0},
        }
        
        # 初始化其他分类变量的中文映射
        self.init_chinese_mappings()
        
        # 创建主界面
        self.create_main_interface()
    
    def init_chinese_mappings(self):
        """初始化其他分类变量的中文映射"""
        # 这里需要根据实际情况填写英文到中文的映射
        # 格式: {特征名: {英文选项: 中文选项}}
        self.other_categorical_mappings = {
            # 示例:
            # 'feature_name': {
            #     'english_option1': '中文选项1',
            #     'english_option2': '中文选项2',
            # }
            
        }
    
    def load_model(self):
        """加载预测模型"""
        try:
            self.predictor = DentalHealthPredictor()
            model_file = "../advanced_dental_model.pkl"
            if os.path.exists(model_file):
                self.predictor.load_model(model_file)
                print("模型加载成功")
            else:
                messagebox.showwarning("警告", "未找到模型文件，请先训练模型")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型时出错: {str(e)}")
    
    def create_main_interface(self):
        """创建主界面"""
        # 主容器设置
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        # 标题区域设置
        title_frame = tk.Frame(main_container, bg=self.bg_color)
        title_frame.pack(pady=30)
        
        # 主标题标签
        title_label = ttk.Label(title_frame, text="口腔健康风险预测系统", style="Title.TLabel")
        title_label.pack()
        
        # 副标题标签
        subtitle_label = ttk.Label(title_frame, text="基于XGBoost的智能预测", style="Subtitle.TLabel")
        subtitle_label.pack(pady=15)
        
        # 按钮区域设置
        button_frame = tk.Frame(main_container, bg=self.bg_color)
        button_frame.pack(pady=40)
        
        # 使用网格布局使按钮对称
        predict_button = ttk.Button(button_frame, text="进行预测", command=self.show_prediction_dialog, style="Primary.TButton", width=20)
        predict_button.grid(row=0, column=0, padx=20, pady=15)
        
        about_caries_button = ttk.Button(button_frame, text="关于龋齿", command=self.show_about_caries, style="Primary.TButton", width=20)
        about_caries_button.grid(row=1, column=0, padx=20, pady=15)
        
        help_button = ttk.Button(button_frame, text="帮助", command=self.show_help, style="Primary.TButton", width=20)
        help_button.grid(row=2, column=0, padx=20, pady=15)
        
        exit_button = ttk.Button(button_frame, text="退出应用", command=self.root.quit, style="Primary.TButton", width=20)
        exit_button.grid(row=3, column=0, padx=20, pady=15)
        
        # 状态栏设置
        status_frame = tk.Frame(self.root, bg=self.secondary_color, height=40)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        status_label = ttk.Label(status_frame, text="就绪", style="TLabel")
        status_label.pack(side=tk.LEFT, padx=15)
    
    def show_prediction_dialog(self):
        """显示预测对话框"""
        if self.predictor is None or self.predictor.model is None:
            messagebox.showerror("错误", "模型未加载，无法进行预测")
            return
        
        # 创建预测窗口
        pred_window = tk.Toplevel(self.root)
        pred_window.title("口腔健康风险预测")
        pred_window.geometry("1400x800")
        pred_window.configure(bg=self.bg_color)
        
        # 创建滚动区域
        canvas = tk.Canvas(pred_window, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(pred_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # 配置滚动区域
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=1380)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 标题设置
        title_label = ttk.Label(scrollable_frame, text="请输入特征值", style="Title.TLabel")
        title_label.pack(pady=20)
        
        # 创建输入字段容器
        input_frame = ttk.Frame(scrollable_frame)
        input_frame.pack(pady=15, padx=20, fill=tk.BOTH)
        
        # 获取特征列表
        excluded_features = ['oral_hygiene_score', 'sweet_hygiene_interaction', 'parent_edu_hygiene', 'check_to_decsyed_interval']
        features = [f for f in self.predictor.features if f not in excluded_features]
        input_values = {}
        
        # 创建输入字段
        for i, feature in enumerate(features):
            row = i // 2
            col = i % 2
            
            frame = ttk.Frame(input_frame)
            frame.grid(row=row, column=col, padx=30, pady=20, sticky="w")
            
            feature_name_cn = self.feature_names_cn.get(feature, feature)
            label = ttk.Label(frame, text=f"{feature_name_cn}:", style="TLabel", width=22)
            label.pack(side=tk.LEFT, padx=5)
            
            if feature in self.categorical_values or feature in self.predictor.label_encoders:
                if feature in self.categorical_values:
                    values = list(self.categorical_values[feature].keys())
                else:
                    if feature in self.other_categorical_mappings:
                        chinese_mapping = self.other_categorical_mappings[feature]
                        original_values = list(self.predictor.label_encoders[feature].classes_)
                        values = [chinese_mapping.get(val, val) for val in original_values]
                        self.chinese_to_english = {v: k for k, v in chinese_mapping.items()}
                    else:
                        values = list(self.predictor.label_encoders[feature].classes_)
                
                var = tk.StringVar()
                combo = RoundedCombobox(frame, textvariable=var, values=values, width=18, state="readonly")
                combo.pack(side=tk.LEFT, padx=5)
                input_values[feature] = var
            else:
                var = tk.StringVar()
                entry = RoundedEntry(frame, textvariable=var, width=18)
                entry.pack(side=tk.LEFT, padx=5)
                input_values[feature] = var
        
        # 预测按钮区域
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(pady=20)
        
        predict_button = ttk.Button(button_frame, text="预测", command=lambda: self.predict(input_values, pred_window), 
                                   style="Primary.TButton", width=15)
        predict_button.pack(padx=10)
        
        # 设置滚动区域
        canvas.pack(side="left", fill="both", expand=True, padx=10)
        scrollbar.pack(side="right", fill="y")
    
    def predict(self, input_values, window):
        """执行预测并显示结果"""
        try:
            # 收集输入值
            input_data = {}
            for feature, var in input_values.items():
                value = var.get()
                print(f"处理特征 {feature}, 原始值: {value}")  # 调试信息
                
                if not value:
                    # 如果用户没有输入值，使用默认值
                    if feature in self.predictor.label_encoders:
                        # 对于分类变量，使用最常见的类别
                        # 获取训练数据中最常见的类别
                        most_common_class = self.predictor.label_encoders[feature].classes_[0]
                        print(f"  使用最常见的类别作为默认值: {most_common_class}")  # 调试信息
                        input_data[feature] = self.predictor.label_encoders[feature].transform([most_common_class])[0]
                    else:
                        # 对于连续变量，使用中位数或平均值
                        # 这里我们暂时使用0，但理想情况下应该使用训练数据的中位数或平均值
                        print(f"  使用默认连续值: 0.0")  # 调试信息
                        input_data[feature] = 0.0
                else:
                    # 处理分类变量
                    if feature in self.categorical_values or feature in self.predictor.label_encoders:
                        try:
                            # 将中文选项转换为对应的数值
                            if feature in self.categorical_values:
                                input_data[feature] = self.categorical_values[feature][value]
                                print(f"  从categorical_values转换: {value} -> {input_data[feature]}")  # 调试信息
                            else:
                                # 如果不在categorical_values中，尝试使用label_encoders
                                # 检查是否需要将中文转换回英文
                                if hasattr(self, 'chinese_to_english') and value in self.chinese_to_english:
                                    # 将中文选项转换回英文
                                    english_value = self.chinese_to_english[value]
                                    input_data[feature] = self.predictor.label_encoders[feature].transform([english_value])[0]
                                    print(f"  从chinese_to_english转换: {value} -> {english_value} -> {input_data[feature]}")  # 调试信息
                                else:
                                    # 直接使用label_encoders转换
                                    input_data[feature] = self.predictor.label_encoders[feature].transform([value])[0]
                                    print(f"  直接使用label_encoders转换: {value} -> {input_data[feature]}")  # 调试信息
                        except Exception as e:
                            # 如果转换失败，使用0作为默认值
                            print(f"  转换失败: {str(e)}")  # 调试信息
                            input_data[feature] = 0
                    else:
                        # 处理数值变量
                        try:
                            input_data[feature] = float(value)
                            print(f"  转换为浮点数: {value} -> {input_data[feature]}")  # 调试信息
                        except Exception as e:
                            # 如果值无效，使用0
                            print(f"  转换为浮点数失败: {str(e)}")  # 调试信息
                            input_data[feature] = 0.0
            
            # 打印最终的输入数据
            print("\n最终的输入数据:")
            for feature, value in input_data.items():
                print(f"{feature}: {value}")
            
            # 进行预测
            try:
                # 准备输入数据
                input_df = pd.DataFrame([input_data])
                
                # 打印输入数据，用于调试
                print("输入数据:")
                print(input_df)
                
                # 处理缺失值
                if hasattr(self.predictor, 'imputer') and self.predictor.imputer is not None:
                    input_df = pd.DataFrame(
                        self.predictor.imputer.transform(input_df),
                        columns=input_df.columns
                    )
                
                # 确保所有列都是数值类型
                for col in input_df.columns:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                
                # 打印处理后的数据，用于调试
                print("处理后的数据:")
                print(input_df)
                
                # 检查模型是否已加载
                if self.predictor.model is None:
                    messagebox.showerror("错误", "模型未加载，请先加载模型")
                    return
                
                # 预测概率
                prob = self.predictor.model.predict_proba(input_df)[0, 1]
                
                # 打印预测概率，用于调试
                print(f"预测概率: {prob}")
                
                # 根据概率确定风险等级
                if prob <= 0.3:  # 低风险阈值
                    risk = "低风险"
                elif prob <= 0.7:  # 中风险阈值
                    risk = "中风险"
                else:
                    risk = "高风险"
                
                # 生成解释和个性化建议
                explanation = self.predictor.generate_explanation(input_df, prob)
                
                # 打印个性化建议，用于调试
                print("\n生成的个性化建议:")
                print(explanation)
                
                result = {
                    'probability': prob,
                    'risk_level': risk,
                    'explanation': explanation
                }
                
                # 显示结果
                self.show_prediction_result(result, window)
                
            except Exception as e:
                messagebox.showerror("错误", f"预测过程中出错: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
        except Exception as e:
            messagebox.showerror("错误", f"处理输入数据时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def generate_explanation(self, input_df, probability):
        """生成预测解释和建议"""
        # 获取特征重要性
        try:
            if hasattr(self.predictor.model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': input_df.columns,
                    'importance': self.predictor.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # 获取前三个最重要的特征
                top_features = importance.head(3)['feature'].tolist()
                
                # 生成解释
                explanation = f"根据您的输入数据，系统预测您患龋齿的概率为{probability:.2%}。\n\n"
                explanation += "对您风险贡献最大的三个因素是：\n"
                
                for i, feature in enumerate(top_features, 1):
                    value = input_df[feature].iloc[0]
                    # 使用中文特征名称
                    feature_name_cn = self.feature_names_cn.get(feature, feature)
                    explanation += f"{i}. {feature_name_cn}: {value}\n"
                
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
            else:
                return f"根据您的输入数据，系统预测您患龋齿的概率为{probability:.2%}。\n\n建议保持良好的口腔卫生习惯，定期进行口腔检查。"
        except Exception as e:
            print(f"生成解释时出错: {str(e)}")
            return f"根据您的输入数据，系统预测您患龋齿的概率为{probability:.2%}。\n\n建议保持良好的口腔卫生习惯，定期进行口腔检查。"
    
    def show_prediction_result(self, result, parent_window):
        """显示预测结果和建议"""
        # 创建结果窗口
        result_window = tk.Toplevel(parent_window)
        result_window.title("预测结果")
        result_window.geometry("800x600")  # 增加窗口大小以容纳更多内容
        result_window.configure(bg=self.bg_color)
        
        # 标题
        title_label = ttk.Label(result_window, text="预测结果", style="Title.TLabel")
        title_label.pack(pady=20)
        
        # 风险等级和患病概率容器
        info_container = tk.Frame(result_window, bg=self.bg_color)
        info_container.pack(pady=15, padx=30, fill=tk.X)
        
        # 风险等级
        risk_frame = ttk.Frame(info_container)
        risk_frame.pack(side=tk.LEFT, expand=True, padx=20)
        
        risk_label = ttk.Label(risk_frame, text="风险等级:", style="TLabel")
        risk_label.pack(pady=5)
        
        risk_value = ttk.Label(risk_frame, text=result.get('risk_level', '未知'), style="Result.TLabel")
        risk_value.pack(pady=5)
        
        # 分隔线
        separator = ttk.Separator(info_container, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=20)
        
        # 患病概率
        prob_frame = ttk.Frame(info_container)
        prob_frame.pack(side=tk.LEFT, expand=True, padx=20)
        
        prob_label = ttk.Label(prob_frame, text="患病概率:", style="TLabel")
        prob_label.pack(pady=5)
        
        prob_value = ttk.Label(prob_frame, text=f"{result.get('probability', 0):.2%}", style="Result.TLabel")
        prob_value.pack(pady=5)
        
        # 解释和建议
        explanation_frame = ttk.Frame(result_window)
        explanation_frame.pack(pady=15, padx=30, fill=tk.BOTH, expand=True)
        
        explanation_label = ttk.Label(explanation_frame, text="个性化建议:", style="Subtitle.TLabel")
        explanation_label.pack(anchor=tk.W, pady=10)
        
        # 创建带滚动条的文本框
        text_container = tk.Frame(explanation_frame, bg=self.bg_color)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        explanation_text = tk.Text(text_container, wrap=tk.WORD, width=70, height=15, 
                                  font=("微软雅黑", 11), bg="#ffffff", fg=self.text_color,
                                  relief=tk.FLAT, padx=15, pady=15)
        explanation_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_container, orient=tk.VERTICAL, command=explanation_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        explanation_text.config(yscrollcommand=scrollbar.set)
        
        # 获取并显示个性化建议
        explanation = result.get('explanation', '无解释')
        explanation_text.insert(tk.END, explanation)
        explanation_text.config(state=tk.DISABLED)
        
        # 关闭按钮
        close_button = ttk.Button(result_window, text="关闭", command=result_window.destroy, 
                                 style="Primary.TButton", width=15)
        close_button.pack(pady=20)
    
    def show_help(self):
        """显示帮助信息"""
        help_window = tk.Toplevel(self.root)
        help_window.title("帮助")
        help_window.geometry("600x500")
        help_window.configure(bg=self.bg_color)
        
        # 标题
        title_label = ttk.Label(help_window, text="关于本系统", style="Title.TLabel")
        title_label.pack(pady=30)
        
        # 帮助内容
        help_text = """
口腔健康风险预测系统

版本: 1.0
作者: 陈雨航 杜锦程 赵文洲（排名不分先后）

本系统基于XGBoost算法，通过分析用户的口腔健康相关特征，
预测用户患龋齿的风险等级，并提供相应的健康建议。
预测结果仅供参考，如有不适请线下就医！

使用方法:
1. 点击"进行预测"按钮
2. 填写特征值，可以留空，系统会使用默认值
3. 点击"预测"按钮获取结果

注意事项:
- 数值字段必须输入数字
- 分类字段必须从下拉菜单中选择

如有问题，请联系技术支持。
作者邮箱：919372267@qq.com
        """
        
        # 创建带滚动条的文本框
        text_container = tk.Frame(help_window, bg=self.bg_color)
        text_container.pack(pady=20, padx=30, fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(text_container, wrap=tk.WORD, width=50, height=15, 
                             font=("微软雅黑", 11), bg="#ffffff", fg=self.text_color,
                             relief=tk.FLAT, padx=15, pady=15)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_container, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.config(yscrollcommand=scrollbar.set)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        
        # 关闭按钮
        close_button = ttk.Button(help_window, text="关闭", command=help_window.destroy, 
                                 style="Primary.TButton", width=15)
        close_button.pack(pady=30)

    def show_about_caries(self):
        """显示关于龋齿的介绍界面"""
        about_window = tk.Toplevel(self.root)
        about_window.title("关于龋齿")
        about_window.geometry("800x600")
        about_window.configure(bg=self.bg_color)
        
        # 标题
        title_label = ttk.Label(about_window, text="龋齿知识介绍", style="Title.TLabel")
        title_label.pack(pady=30)
        
        # 创建带滚动条的文本框
        text_container = tk.Frame(about_window, bg=self.bg_color)
        text_container.pack(pady=20, padx=30, fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(text_container, wrap=tk.WORD, width=70, height=20, 
                            font=("微软雅黑", 11), bg="#ffffff", fg=self.text_color,
                            relief=tk.FLAT, padx=15, pady=15)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_container, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # 添加龋齿相关知识
        caries_info = """
什么是龋齿？
龋齿，俗称"虫牙"，是一种常见的口腔疾病。它是由口腔中的细菌分解食物中的糖分产生酸，这些酸会腐蚀牙齿的硬组织，导致牙齿出现缺损。

龋齿的危害：
1. 疼痛：龋齿会导致牙齿疼痛，影响日常生活
2. 感染：严重时可能引起牙髓炎、根尖周炎等
3. 影响咀嚼：影响食物的咀嚼和消化
4. 影响美观：影响面部美观和自信心
5. 影响发音：可能影响语言发音

龋齿的预防：
1. 保持良好的口腔卫生习惯
   - 每天至少刷牙两次
   - 使用牙线清洁牙缝
   - 定期使用漱口水

2. 控制饮食
   - 减少糖分摄入
   - 避免频繁食用甜食
   - 注意饮食均衡

3. 定期检查
   - 每半年进行一次口腔检查
   - 及时发现和处理问题
   - 进行预防性治疗

4. 其他预防措施
   - 使用含氟牙膏
   - 进行窝沟封闭
   - 定期涂氟

龋齿的治疗：
1. 早期治疗
   - 补牙
   - 窝沟封闭
   - 涂氟治疗

2. 中期治疗
   - 根管治疗
   - 牙冠修复

3. 晚期治疗
   - 拔牙
   - 种植牙
   - 假牙修复

注意事项：
1. 发现龋齿要及时治疗
2. 保持良好的口腔卫生习惯
3. 定期进行口腔检查
4. 注意饮食健康
5. 避免不良习惯（如咬硬物、磨牙等）

温馨提示：
预防胜于治疗，保持良好的口腔卫生习惯是预防龋齿的关键。如果发现牙齿有问题，请及时就医，不要拖延。
"""
        text_widget.insert(tk.END, caries_info)
        text_widget.config(state=tk.DISABLED)
        
        # 关闭按钮
        close_button = ttk.Button(about_window, text="关闭", command=about_window.destroy, 
                                style="Primary.TButton", width=15)
        close_button.pack(pady=30)

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalHealthApp(root)
    root.mainloop() 