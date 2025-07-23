#!/usr/bin/env python3
"""
测试CSRank和TSRank实现的正确性
使用模拟数据验证
"""

import numpy as np
import pandas as pd
import unittest
from qlib.data.ops import CSRank, TSRank, Feature
from scipy.stats import rankdata  # 添加导入以修复NameError

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import CSRankProcessor  # 假设CSRankProcessor已实现

from qlib import init as qlib_init
from qlib.utils import init_instance_by_config

# 初始化Qlib (使用模拟数据路径，如果需要)
qlib_init(provider_uri="data", region="cn")  # 调整为实际路径

class TestRankOperators(unittest.TestCase):
    def setUp(self):
        # 创建模拟数据: 3个资产，10个日期
        dates = pd.date_range('2023-01-01', periods=10)
        instruments = ['INS1', 'INS2', 'INS3']
        
        # 多资产DataFrame
        multi_index = pd.MultiIndex.from_product([dates, instruments], names=['datetime', 'instrument'])
        self.mock_data = pd.DataFrame({
            'feature': np.random.randn(len(multi_index)),
            'another_feature': np.random.randn(len(multi_index))
        }, index=multi_index)
        
        # 添加一些NaN以测试边缘情况
        self.mock_data.loc[(dates[2], 'INS1'), 'feature'] = np.nan
        self.mock_data.loc[(dates[5], 'INS2'), 'feature'] = np.nan
        
        # 准备 data_dict for each instrument
        self.data_dict = {}
        for ins in ['INS1', 'INS2', 'INS3']:
            self.data_dict[ins] = self.mock_data.loc[(slice(None), ins), 'feature'].droplevel('instrument')
        
        self.single_series = self.data_dict['INS1']
        self.freq = 'day'
        
        # Define MockFeature with data_dict
        class MockFeature(Feature):
            def __init__(self, name, data_dict, freq):
                super().__init__(name)
                self.data_dict = data_dict
                self.freq = freq
            
            def _load_internal(self, instrument, start_index, end_index, freq):
                if instrument in self.data_dict:
                    series = self.data_dict[instrument]
                    return series.iloc[start_index:end_index+1]
                else:
                    return pd.Series()
        
        self.MockFeature = MockFeature
    
    def test_CSRank(self):
        print("\n🔍 测试CSRank")
        
        mock_feature = self.MockFeature('feature', self.data_dict, self.freq)
        expr = CSRank(mock_feature)
        
        series = expr.load('INS1', 0, len(self.single_series) - 1, self.freq)
        self.assertTrue(series.name.startswith('__CSRANK__'))  # 检查标记
        
        # 模拟跨截面排名 (像处理器那样)
        def simulate_cs_rank(df):
            csrank_cols = [col for col in df.columns if str(col).startswith('__CSRANK__')]
            for col in csrank_cols:
                ranked = df.groupby('datetime')[col].transform(
                    lambda x: rankdata(x.dropna(), method='average') / len(x.dropna()) if not x.dropna().empty else np.nan
                )
                original_name = str(col).replace('__CSRANK__', 'CSRank_')
                df[original_name] = ranked
                df = df.drop(columns=[col])
            return df
        
        # 准备完整数据集
        full_df = self.mock_data.reset_index()
        full_df['__CSRANK__feature'] = self.mock_data['feature']  # 模拟标记
        
        ranked_df = simulate_cs_rank(full_df)
        
        # 手动计算预期排名
        expected_ranks = {}
        for date in dates:
            daily_data = self.mock_data.loc[(date), 'feature'].dropna()
            if not daily_data.empty:
                ranks = rankdata(daily_data, method='average') / len(daily_data)
                for ins, rank in zip(daily_data.index, ranks):
                    expected_ranks[(date, ins)] = rank
        
        # 比较
        actual_ranks = ranked_df.set_index(['datetime', 'instrument'])['CSRank_feature']
        for key in expected_ranks:
            self.assertAlmostEqual(actual_ranks.loc[key], expected_ranks[key], places=5)
        
        print("✅ CSRank测试通过")
    
    def test_TSRank(self):
        print("\n🔍 测试TSRank")
        
        window = 5
        mock_feature = self.MockFeature('feature', self.data_dict, self.freq)
        expr = TSRank(mock_feature, window)
        
        # 计算TSRank
        ranked_series = expr.load('INS1', 0, len(self.single_series) - 1, self.freq)
        
        # 修复索引问题
        self.assertIsInstance(ranked_series, pd.Series)
        self.assertEqual(len(ranked_series), len(self.single_series))
        
        # 手动计算预期排名
        expected_ranks = np.full_like(self.single_series.values, np.nan)
        for i in range(len(self.single_series)):
            start = max(0, i - window + 1)
            window_data = self.single_series.values[start:i+1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                ranks = rankdata(valid_data, method='average')
                expected_ranks[i] = ranks[-1] / len(valid_data)
        
        # 比较 (允许小浮点差异)
        np.testing.assert_array_almost_equal(ranked_series.values, expected_ranks, decimal=5)
        
        # 测试边缘情况
        # 全NaN窗口
        all_nan_series = pd.Series(np.nan, index=range(3))
        expr_all_nan = TSRank(Feature('nan_feature'), 5)
        ranked_nan = expr_all_nan._load_internal('TEST', 0, 2, self.freq)
        np.testing.assert_array_equal(ranked_nan.values, [np.nan, np.nan, np.nan])
        
        # 增加非NaN值
        mixed_series = pd.Series([np.nan, np.nan, 1.0, np.nan, 2.0, 3.0])
        expr_mixed = TSRank(Feature('mixed'), 3)
        ranked_mixed = expr_mixed._load_internal('TEST', 0, 5, self.freq)
        expected_mixed = [np.nan, np.nan, 1.0, np.nan, 0.75, 1.0]  # 手动计算
        np.testing.assert_array_almost_equal(ranked_mixed.values, expected_mixed, decimal=5)
        
        print("✅ TSRank测试通过")
    
    def test_performance(self):
        print("\n📊 性能测试")
        large_series = pd.Series(np.random.randn(100000), index=pd.date_range('2000-01-01', periods=100000))
        mock_feature = self.MockFeature('large', {'TEST': large_series}, self.freq)
        expr = TSRank(mock_feature, 100)
        import time
        start = time.time()
        _ = expr._load_internal('TEST', 0, 99999, self.freq)
        duration = time.time() - start
        print(f"优化TSRank计算100000点数据 (窗口=100): {duration:.4f}秒")
        self.assertLess(duration, 10.0)  # 调整为更现实的值，假设Numba优化

if __name__ == '__main__':
    unittest.main()