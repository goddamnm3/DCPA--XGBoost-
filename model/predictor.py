import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import shap
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class DentalHealthPredictor:
    def __init__(self):
        self.model = None
        self.imputer = None
        self.label_encoders = {}
        self.features = []
        self.calibrator = None
        self.shap_explainer = None
        self.training_data = None
        self.target_variable = None
        self.original_features = []
        self.feature_statistics = {}

    def load_model(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件 {file_path} 不存在")

        model_data = joblib.load(file_path)
        self.model = model_data['model']
        self.imputer = model_data['imputer']
        self.label_encoders = model_data['label_encoders']
        self.features = model_data['features']
        self.calibrator = model_data.get('calibrator', None)
        print(f"\n已从 {file_path} 加载模型")

    def save_model(self, file_path):
        model_data = {
            'model': self.model,
            'imputer': self.imputer,
            'label_encoders': self.label_encoders,
            'features': self.features,
            'calibrator': self.calibrator
        }
        joblib.dump(model_data, file_path)
        print(f"\n模型已保存到 {file_path}")

    def generate_explanation(self, input_df, probability):
        explanation = f"根据您的输入数据，系统预测您患龋齿的概率为{probability:.2%}。\n\n"
        
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
        try:
            input_df = pd.DataFrame([input_data])
            
            for feature in input_df.columns:
                if feature in self.label_encoders:
                    if not pd.api.types.is_numeric_dtype(input_df[feature]):
                        input_df[feature] = input_df[feature].astype(float)
                else:
                    input_df[feature] = input_df[feature].astype(float)
            
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
            
            missing_features = set(self.features) - set(input_df.columns)
            if missing_features:
                print(f"警告：以下特征在输入数据中缺失，将使用默认值：{missing_features}")
                for feature in missing_features:
                    if feature in self.label_encoders:
                        input_df[feature] = float(self.label_encoders[feature].transform([self.label_encoders[feature].classes_[0]])[0])
                    else:
                        input_df[feature] = 0.0
            
            input_df = input_df[self.features]
            
            input_df = pd.DataFrame(
                self.imputer.transform(input_df),
                columns=input_df.columns
            )
            
            prob = self.model.predict_proba(input_df)[0, 1]
            
            if prob <= 0.3:
                risk = "低风险"
            elif prob <= 0.7:
                risk = "中风险"
            else:
                risk = "高风险"
            
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(input_df)
                
                feature_importance = pd.DataFrame({
                    '特征': self.features,
                    'SHAP值': np.abs(shap_values[0])
                }).sort_values('SHAP值', ascending=False)
                
                top_3_features = feature_importance.head(3)
                
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
                
                specific_suggestions = []
                for _, row in top_3_features.iterrows():
                    feature = row['特征']
                    value = input_data.get(feature, 0)
                    
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
                
                explanation = f"根据您的输入数据，系统预测您患龋齿的概率为{prob:.2%}。\n\n"
                
                explanation += "通用建议：\n"
                for i, suggestion in enumerate(general_suggestions, 1):
                    explanation += f"{i}. {suggestion}\n"
                
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
            return {
                'probability': 0.3,
                'risk_level': "低风险",
                'explanation': "由于数据处理过程中出现错误，系统返回默认的低风险结果。请检查输入数据的格式是否正确。"
            } 