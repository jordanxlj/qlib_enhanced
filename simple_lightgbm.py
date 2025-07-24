import multiprocessing  # 用于Windows多进程修复
import pickle  # 新增：用于手动保存模型

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.data.dataset import DatasetH, DataHandlerLP  # 导入以检查类型
from qlib.contrib.data.handler import Alpha158  # 直接导入类检查
from qlib.contrib.model.gbdt import LGBModel
from qlib.data import D  # 新增：用于加载基准数据
import pandas as pd
import numpy as np  # 用于手动预测和计算
import scipy.stats as stats  # 新增：用于计算相关性（IC）

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

    # 新增：IC分析（Information Coefficient）
    print("\n开始IC分析...")
    # 手动提取测试集实际标签（DK_L: 学习数据，包含标签）
    test_label_df = handler.fetch(data_key=DataHandlerLP.DK_L)  # 获取带标签的数据
    test_label_df = test_label_df.loc[(test_label_df.index.get_level_values('datetime') >= test_start) &
                                      (test_label_df.index.get_level_values('datetime') <= test_end)]

    # 假设标签列是'LABEL0'（实际回报）
    actual = test_label_df.get('LABEL0', pd.Series())  # 获取实际回报Series

    # 对齐pred和actual（确保相同索引）
    common_index = pred.index.intersection(actual.index)
    pred_aligned = pred.loc[common_index]
    actual_aligned = actual.loc[common_index]

    # 计算整体IC（Pearson和Spearman/Rank IC）
    ic_pearson = pred_aligned.corr(actual_aligned, method='pearson')
    ic_spearman = pred_aligned.corr(actual_aligned, method='spearman')  # Rank IC

    print(f"整体Pearson IC: {ic_pearson:.4f}")
    print(f"整体Spearman (Rank) IC: {ic_spearman:.4f}")

    # （可选）逐日IC
    daily_ic = []
    for date in pred_aligned.index.get_level_values('datetime').unique():
        daily_pred = pred_aligned.xs(date, level='datetime')
        daily_actual = actual_aligned.xs(date, level='datetime')
        if len(daily_pred) > 1 and len(daily_actual) > 1:
            daily_ic.append(daily_pred.corr(daily_actual, method='spearman'))
    mean_daily_ic = np.mean(daily_ic)
    print(f"平均逐日Rank IC: {mean_daily_ic:.4f}")

    # 新增：与基准比较（例如CSI300指数）
    print("\n开始与基准比较...")
    benchmark = 'SH000300'  # 基准：CSI300
    bench_fields = ['Ref($close, -1) / $close - 1']  # 计算基准每日回报
    bench_data = D.features([benchmark], bench_fields, start_time=test_start, end_time=test_end, freq='day')
    bench_data.columns = ['bench_return']
    bench_returns = bench_data['bench_return'].droplevel('instrument')  # 转换为Series，索引为datetime

    # 简单策略回报模拟：基于pred的等权重Top50股票（示例）
    # 逐日计算策略回报
    strategy_returns = []
    dates = pred.index.get_level_values('datetime').unique()
    for date in dates:
        daily_pred = pred.xs(date, level='datetime')
        # 选Top50（按pred降序）
        topk = daily_pred.nlargest(50).index
        # 假设等权重，实际回报从actual（或加载$close计算）
        daily_actual = actual.xs(date, level='datetime').reindex(topk)
        if not daily_actual.empty:
            strategy_returns.append(daily_actual.mean())  # 等权重平均回报
        else:
            strategy_returns.append(0.0)
    strategy_returns = pd.Series(strategy_returns, index=dates)

    # 计算累计回报
    cum_strategy = (1 + strategy_returns).cumprod()
    cum_bench = (1 + bench_returns).cumprod()

    # 打印比较
    print("策略累计回报（测试期末）:", cum_strategy.iloc[-1])
    print("基准累计回报（测试期末）:", cum_bench.iloc[-1])
    print("策略年化回报:", (cum_strategy.iloc[-1] ** (252 / len(strategy_returns)) - 1))  # 假设252交易日
    print("基准年化回报:", (cum_bench.iloc[-1] ** (252 / len(bench_returns)) - 1))

    # （可选）保存比较数据
    comparison_df = pd.DataFrame({'strategy_cum': cum_strategy, 'bench_cum': cum_bench})
    comparison_df.to_csv('return_comparison.csv')

    # 手动保存模型或预测（使用pickle）
    print("保存模型和预测...")
    with open('lightgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)  # 保存整个Qlib模型对象
    pred.to_pickle('predictions.pkl')  # 保存预测
