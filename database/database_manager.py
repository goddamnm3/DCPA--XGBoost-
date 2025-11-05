import mysql.connector
from datetime import datetime
import pandas as pd

class DatabaseManager:
    def __init__(self):
        """初始化数据库连接"""
        try:
            self.conn = mysql.connector.connect(
                host="localhost",
                user="root",  # 替换为你的MySQL用户名
                password="123456",  # 替换为你的MySQL密码
                database="dental_health"  # 替换为你的数据库名
            )
            self.cursor = self.conn.cursor()
            self.create_tables()
            print("数据库连接成功")
        except mysql.connector.Error as err:
            print(f"数据库连接错误: {err}")
            raise

    def create_tables(self):
        """创建必要的数据表"""
        try:
            # 创建预测记录表
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_records (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    prediction_time DATETIME,
                    risk_level VARCHAR(20),
                    probability FLOAT,
                    age INT,
                    gender VARCHAR(10),
                    height FLOAT,
                    weight FLOAT,
                    brush_time FLOAT,
                    brush_counter INT,
                    milk_time VARCHAR(50),
                    oral_hygiene_score FLOAT,
                    sweet_hygiene_interaction FLOAT,
                    parent_edu_hygiene FLOAT,
                    check_to_decsyed_interval FLOAT,
                    prediction_features TEXT,
                    suggestions TEXT
                )
            """)
            
            # 创建用户反馈表
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    prediction_id INT,
                    feedback_time DATETIME,
                    feedback_content TEXT,
                    FOREIGN KEY (prediction_id) REFERENCES prediction_records(id)
                )
            """)
            
            self.conn.commit()
            print("数据表创建成功")
        except mysql.connector.Error as err:
            print(f"创建数据表错误: {err}")
            raise

    def save_prediction(self, prediction_data):
        """
        保存预测记录
        prediction_data: 包含预测结果和输入特征的字典
        """
        try:
            sql = """
                INSERT INTO prediction_records (
                    prediction_time, risk_level, probability,
                    age, gender, height, weight, brush_time,
                    brush_counter, milk_time, oral_hygiene_score,
                    sweet_hygiene_interaction, parent_edu_hygiene,
                    check_to_decsyed_interval, prediction_features,
                    suggestions
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            values = (
                datetime.now(),
                prediction_data.get('risk_level'),
                prediction_data.get('probability'),
                prediction_data.get('age'),
                prediction_data.get('gender'),
                prediction_data.get('height'),
                prediction_data.get('weight'),
                prediction_data.get('brush_time'),
                prediction_data.get('brush_counter'),
                prediction_data.get('milk_time'),
                prediction_data.get('oral_hygiene_score'),
                prediction_data.get('sweet_hygiene_interaction'),
                prediction_data.get('parent_edu_hygiene'),
                prediction_data.get('check_to_decsyed_interval'),
                str(prediction_data.get('features', {})),
                str(prediction_data.get('suggestions', []))
            )
            
            self.cursor.execute(sql, values)
            self.conn.commit()
            print("预测记录保存成功")
            return self.cursor.lastrowid
        except mysql.connector.Error as err:
            print(f"保存预测记录错误: {err}")
            raise

    def save_feedback(self, prediction_id, feedback_content):
        """保存用户反馈"""
        try:
            sql = """
                INSERT INTO user_feedback (
                    prediction_id, feedback_time, feedback_content
                ) VALUES (%s, %s, %s)
            """
            
            values = (
                prediction_id,
                datetime.now(),
                feedback_content
            )
            
            self.cursor.execute(sql, values)
            self.conn.commit()
            print("用户反馈保存成功")
        except mysql.connector.Error as err:
            print(f"保存用户反馈错误: {err}")
            raise

    def get_prediction_history(self, limit=100):
        """获取预测历史记录"""
        try:
            sql = """
                SELECT * FROM prediction_records
                ORDER BY prediction_time DESC
                LIMIT %s
            """
            
            self.cursor.execute(sql, (limit,))
            columns = [desc[0] for desc in self.cursor.description]
            results = self.cursor.fetchall()
            
            return pd.DataFrame(results, columns=columns)
        except mysql.connector.Error as err:
            print(f"获取预测历史记录错误: {err}")
            raise

    def get_prediction_statistics(self):
        """获取预测统计数据"""
        try:
            sql = """
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(probability) as avg_probability,
                    COUNT(CASE WHEN risk_level = '高风险' THEN 1 END) as high_risk_count,
                    COUNT(CASE WHEN risk_level = '中风险' THEN 1 END) as medium_risk_count,
                    COUNT(CASE WHEN risk_level = '低风险' THEN 1 END) as low_risk_count
                FROM prediction_records
            """
            
            self.cursor.execute(sql)
            return self.cursor.fetchone()
        except mysql.connector.Error as err:
            print(f"获取预测统计数据错误: {err}")
            raise

    def close(self):
        """关闭数据库连接"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            print("数据库连接已关闭")
        except mysql.connector.Error as err:
            print(f"关闭数据库连接错误: {err}")
            raise

# 使用示例
if __name__ == "__main__":
    try:
        # 创建数据库管理器实例
        db_manager = DatabaseManager()
        
        # 示例：保存预测记录
        prediction_data = {
            'risk_level': '中风险',
            'probability': 0.65,
            'age': 25,
            'gender': '男',
            'height': 175.0,
            'weight': 70.0,
            'brush_time': 3.0,
            'brush_counter': 2,
            'milk_time': '睡前',
            'oral_hygiene_score': 0.8,
            'sweet_hygiene_interaction': 0.6,
            'parent_edu_hygiene': 0.7,
            'check_to_decsyed_interval': 12.0,
            'features': {'feature1': 0.5, 'feature2': 0.3},
            'suggestions': ['建议1', '建议2']
        }
        
        prediction_id = db_manager.save_prediction(prediction_data)
        
        # 示例：保存用户反馈
        db_manager.save_feedback(prediction_id, "预测结果很准确")
        
        # 示例：获取预测历史
        history = db_manager.get_prediction_history()
        print("\n预测历史记录:")
        print(history)
        
        # 示例：获取统计数据
        stats = db_manager.get_prediction_statistics()
        print("\n预测统计数据:")
        print(f"总预测次数: {stats[0]}")
        print(f"平均风险概率: {stats[1]:.2%}")
        print(f"高风险数量: {stats[2]}")
        print(f"中风险数量: {stats[3]}")
        print(f"低风险数量: {stats[4]}")
        
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 关闭数据库连接
        db_manager.close() 