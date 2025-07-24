import multiprocessing  # 用于Windows多进程修复
import pickle  # 新增：用于手动保存模型

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.data.dataset import DatasetH, DataHandlerLP  # 导入以检查类型
from qlib.contrib.data.handler import Alpha158  # 直接导入类检查
from qlib.contrib.model.gbdt import LGBModel
import pandas as pd
import numpy as np  # 用于手动预测

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows多进程修复

    # 初始化Qlib（替换为你的数据路径）
    qlib.init(provider_uri='data', region=REG_CN)

    # 先定义并创建handler配置（最小化：移除processors测试）
    handler_config = {
        'class': 'Alpha158',
        'module_path': 'qlib.contrib.data.handler',
        'kwargs': {
            'instruments': 'csi300',  # 股票池
            'start_time': '2008-01-01',
            'end_time': '2020-08-01',
            'fit_start_time': '2008-01-01',
            'fit_end_time': '2014-12-31',
            # 先注释processors，测试最小配置
            # 'infer_processors': [{'class': 'ZScoreNorm'}, {'class': 'Fillna'}],
            # 'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}]
        }
    }

    # 创建handler实例
    handler = init_instance_by_config(handler_config)

    # 调试：打印handler类型和样本数据
    print(f"Type of handler: {type(handler)}")  # 预期: <class 'qlib.contrib.data.handler.Alpha158'>
    if not isinstance(handler, DataHandlerLP):
        raise ValueError("Handler不是DataHandlerLP类型，请检查Qlib安装或配置！")
    print("Handler样本数据：")
    print(handler.fetch().head())  # 打印handler数据样本，确保不是空

    # 现在创建dataset，传入handler和segments
    dataset = DatasetH(handler=handler, segments={
        'train': ('2008-01-01', '2014-12-31'),
        'valid': ('2015-01-01', '2016-12-31'),
        'test': ('2017-01-01', '2020-08-01')
    })

    # 调试：打印dataset类型
    print(f"Type of dataset: {type(dataset)}")  # 预期: <class 'qlib.data.dataset.DatasetH'>
    if not isinstance(dataset, DatasetH):
        raise ValueError("Dataset不是DatasetH类型，请检查配置或Qlib版本！")

    # 定义模型配置
    model_config = {
        'class': 'LGBModel',
        'module_path': 'qlib.contrib.model.gbdt',
        'kwargs': {
            'loss': 'mse',
            'colsample_bytree': 0.8879,
            'learning_rate': 0.2,
            'subsample': 0.8789,
            'lambda_l1': 205.6999,
            'lambda_l2': 580.9768,
            'max_depth': 8,
            'num_leaves': 210,
            'num_threads': 20  # 多线程配置
        }
    }

    # 创建模型实例
    model = init_instance_by_config(model_config)

    # 训练模型
    print("开始训练...")
    model.fit(dataset)  # 这会自动处理数据准备、特征计算和训练

    # 调试：在predict前再次检查类型（如果这里变了，说明fit修改了dataset）
    print(f"Type of dataset after fit: {type(dataset)}")

    # 预测（默认使用workaround绕过prepare问题）
    print("开始预测...")
    # 手动提取测试特征（DK_I: 推理数据，仅特征）
    test_df = handler.fetch(data_key=DataHandlerLP.DK_I)  # 获取所有推理数据
    # 过滤测试时间段（假设索引是MultiIndex with datetime）
    test_start, test_end = pd.Timestamp('2017-01-01'), pd.Timestamp('2020-08-01')
    test_df = test_df.loc[(test_df.index.get_level_values('datetime') >= test_start) &
                          (test_df.index.get_level_values('datetime') <= test_end)]

    # 假设特征列是除标签外的所有列（根据你的handler，排除LABEL0，如果存在）
    feature_cols = [col for col in test_df.columns if col != 'LABEL0']  # 调整为你的标签列（如果有）
    X_test = test_df[feature_cols].values  # 转换为numpy array

    # 使用模型的底层LightGBM预测
    pred_array = model.model.predict(X_test)  # model.model是底层lgb.Booster
    pred = pd.Series(pred_array, index=test_df.index)  # 转换为Series

    # 查看预测结果（可选：保存或分析）
    print("预测结果样本：")
    print(pred.head())

    # 手动保存模型或预测（使用pickle）
    print("保存模型和预测...")
    with open('lightgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)  # 保存整个Qlib模型对象
    pred.to_pickle('predictions.pkl')  # 保存预测