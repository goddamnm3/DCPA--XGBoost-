import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings("ignore")


# 加载数据
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 精简后的参数空间（控制变量）
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

# 将参数转换为贝叶斯搜索格式
param_bayes = {
    'n_estimators': (50, 100),
    'max_depth': (3, 5),
    'learning_rate': (0.01, 0.1, 'log-uniform')
}

# 计算网格搜索总组合数（控制贝叶斯 n_iter）
total_combinations = len(list(ParameterGrid(param_grid)))


# 训练器类
class XGBoostTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self, search_type='grid'):
        if search_type == 'grid':
            search = GridSearchCV(
                estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0),
                param_grid=param_grid,
                cv=3,
                n_jobs=-1
            )
        elif search_type == 'bayes':
            search = BayesSearchCV(
                estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0),
                search_spaces=param_bayes,
                n_iter=total_combinations,
                cv=3,
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError("search_type must be 'grid' or 'bayes'")

        start_time = time.time()
        search.fit(self.X_train, self.y_train)
        end_time = time.time()

        best_model = search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        duration = end_time - start_time
        efficiency_score = accuracy / duration

        print(f"\n=== {search_type.upper()} SEARCH RESULTS ===")
        print(f"Best Parameters: {search.best_params_}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Search Time: {duration:.2f} seconds")
        print(f"Efficiency Score (accuracy/sec): {efficiency_score:.4f}")

        return {
            "method": search_type,
            "best_params": search.best_params_,
            "accuracy": accuracy,
            "train_time": duration,
            "efficiency": efficiency_score
        }


# 主程序运行
trainer = XGBoostTrainer(X_train, y_train, X_test, y_test)

grid_result = trainer.train_model(search_type='grid')
bayes_result = trainer.train_model(search_type='bayes')


# 简洁汇总
print("\n=== SUMMARY COMPARISON ===")
print("{:<10} | {:<10} | {:<10} | {:<10}".format("Method", "Time(s)", "Accuracy", "Efficiency"))
print("-" * 45)
for result in [grid_result, bayes_result]:
    print("{:<10} | {:<10.2f} | {:<10.4f} | {:<10.4f}".format(
        result["method"], result["train_time"], result["accuracy"], result["efficiency"]
    ))
