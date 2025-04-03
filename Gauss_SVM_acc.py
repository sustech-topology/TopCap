import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def parallel_gaussian_svm_cv(csv_file, n_jobs=-1):
    """
    使用并行计算加速高斯SVM五折交叉验证
    
    参数:
        csv_file: CSV文件路径
        n_jobs: 并行任务数（-1表示使用所有CPU核心）
    
    返回:
        交叉验证的平均准确率和各折结果
    """
    try:
        # 1. 读取数据
        data = pd.read_csv(csv_file, header=None)
        X = data.iloc[:, [2, 3]].values  # 第3、4列是特征
        y = data.iloc[:, 4].values       # 第5列是标签

        # 2. 检查是否为二分类
        if len(np.unique(y)) != 2:
            raise ValueError("标签必须是二分类数据！")

        # 3. 创建管道（标准化 + SVM）
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', gamma='scale', random_state=42)
        )

        # 4. 分层五折交叉验证（并行计算）
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X, y,
            cv=skf,
            scoring='accuracy',
            n_jobs=n_jobs  # 并行运行交叉验证
        )

        # 5. 输出结果
        print(f"五折交叉验证准确率: {cv_scores}")
        print(f"平均准确率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

        return {
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'all_scores': cv_scores
        }

    except Exception as e:
        print(f"发生错误: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    

    csv_path = 'D:\\phonetic\\All_dataset\\Sample_TIMIT_noise0_arr.csv'
    print(csv_path)
    results = parallel_gaussian_svm_cv(csv_path, n_jobs=6)