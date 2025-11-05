import pandas as pd
import mysql.connector
from mysql.connector import Error


def create_connection():
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',  # MySQL 主机地址
            database='train_sample',  # 目标数据库名
            user='root',  # 数据库用户名
            password='hang0322'  # 数据库密码
        )

        if connection.is_connected():
            print("连接到数据库成功")
            return connection
    except Error as e:
        print(f"数据库连接错误: {e}")
        return None

def load_data(file_path):
    # 读取 Excel 文件
    df = pd.read_excel(file_path)
    return df

def insert_data(connection, df, table_name):
    cursor = connection.cursor()
    for index, row in df.iterrows():
        # 为每一行数据创建插入 SQL 语句
        sql = f"""
        INSERT INTO {table_name} (id,write_time,gender,age,height,weight,brush_time,brush_method, toothpaste,brush_counter,wash_meal,floss_seq,wash_seq,sweet_seq,sweet_drink_seq,other_snack_seq,dental_seq,first_brush,help_until,first_check,first_decsyed,descsyed_counter,descsyed_cure,milk_time,milk_method,whether_F,F_seq,sealant,systemic_disease,self_brush,parent_edu,parent_job,family_income,parent_health,parent_emphasis,whether_cure,writer_id) 
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """
        cursor.execute(sql, tuple(row))
    connection.commit()
    print(f"数据成功插入到 {table_name} 表中。")

def main():
    # 设置 Excel 文件路径
    file_path = '/dachuang/train_sample.xlsx'
    # 设置 MySQL 表名
    table_name = 'questionare'

    # 连接到数据库
    connection = create_connection()
    if connection is None:
        return

    # 加载 Excel 数据
    df = load_data(file_path)

    # 将数据插入 MySQL
    insert_data(connection, df, table_name)

    # 关闭数据库连接
    connection.close()

if __name__ == "__main__":
    main()
