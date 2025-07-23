import multiprocessing  # 新增：导入multiprocessing以使用freeze_support

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
import pandas as pd

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 新增：Windows修复多进程问题

    # 初始化Qlib（替换为你的数据路径）
    qlib.init(provider_uri='data', region=REG_CN)

    # 定义数据集配置（类似于YAML中的dataset部分）
    dataset_config = {
        'class': 'DatasetH',
        'module_path': 'qlib.data.dataset',
        'kwargs': {
            'handler': {
                'class': 'Alpha158',  # 使用Alpha158特征
                'module_path': 'qlib.contrib.data.handler',
                'kwargs': {
                    'instruments': 'csi300',  # 股票池
                    'start_time': '2008-01-01',
                    'end_time': '2020-08-01',
                    'fit_start_time': '2008-01-01',
                    'fit_end_time': '2014-12-31',
                    'infer_processors': [{'class': 'ZScoreNorm'}, {'class': 'Fillna'}],  # 推理处理器
                    'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}]  # 学习处理器
                }
            },
            'segments': {
                'train': ('2008-01-01', '2014-12-31'),
                'valid': ('2015-01-01', '2016-12-31'),
                'test': ('2017-01-01', '2020-08-01')
            }
        }
    }

    # 创建数据集实例
    dataset = init_instance_by_config(dataset_config)

    # 定义模型配置（类似于YAML中的model部分）
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
            'num_threads': 1  # 多线程配置
        }
    }

    # 创建模型实例
    model = init_instance_by_config(model_config)

    # 训练模型（直接fit，不用工作流）
    print("开始训练...")
    model.fit(dataset)  # 这会自动处理数据准备、特征计算和训练

    # 预测（在测试集上）
    print("开始预测...")
    pred = model.predict(dataset.prepare('test'))  # pred是一个Series，索引为(datetime, instrument)，值为预测回报

    # 查看预测结果（可选：保存或分析）
    print("预测结果样本：")
    print(pred.head())

    # （可选）手动保存模型或预测
    model.save('lightgbm_model.pkl')  # 保存模型
    pred.to_pickle('predictions.pkl')  # 保存预测