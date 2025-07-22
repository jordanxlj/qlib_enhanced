#!/usr/bin/env python3
"""
CSI300 数据性能测试脚本
专门用于测试qlib表达式在CSI300股票池上的性能
"""

import time
import psutil
import numpy as np
import pandas as pd
import yaml
import sys
import gc
import warnings
from typing import List, Dict, Tuple, Optional
import traceback

# 添加qlib路径
sys.path.insert(0, '/d:/code/stock/qlib')

try:
    import qlib
    from qlib.data import D
    from qlib.config import C
    
    # 初始化qlib
    qlib.init(provider_uri="data", region="cn")
    
except ImportError as e:
    print(f"❌ 无法导入qlib: {e}")
    sys.exit(1)

class CSI300PerformanceTester:
    """CSI300性能测试器"""
    
    def __init__(self):
        self.results = []
        
    def get_csi300_instruments(self, sample_size: int = None) -> List[str]:
        """获取CSI300成分股列表"""
        try:
            # 直接使用CSI300成分股列表（更可靠的方法）
            # 由于D.instruments()返回的是配置字典而不是股票列表，我们直接使用已知的CSI300股票
            instruments = self._get_csi300_stock_list()
            
            if sample_size and len(instruments) > sample_size:
                # 随机采样，但保证包含一些知名股票
                import random
                
                # 确保包含一些知名大盘股
                priority_stocks = [
                    "SH600519",  # 贵州茅台
                    "SZ000858",  # 五粮液
                    "SH600036",  # 招商银行
                    "SZ000001",  # 平安银行
                    "SH601318",  # 中国平安
                    "SH600000",  # 浦发银行
                    "SZ000002",  # 万科A
                    "SH601166",  # 兴业银行
                    "SH600276",  # 恒瑞医药
                    "SZ002415",  # 海康威视
                ]
                
                available_priority = [s for s in priority_stocks if s in instruments]
                remaining_slots = max(0, sample_size - len(available_priority))
                
                other_stocks = [s for s in instruments if s not in available_priority]
                random.shuffle(other_stocks)
                
                selected = available_priority + other_stocks[:remaining_slots]
                return selected[:sample_size]
            
            return instruments
            
        except Exception as e:
            print(f"⚠️  无法获取CSI300成分股，使用默认股票池: {e}")
            # 备用股票池
            return [
                "SH600519", "SZ000858", "SH600036", "SZ000001", "SH601318",
                "SH600276", "SZ000002", "SH601166", "SH600000", "SZ002415"
            ]
    
    def _get_csi300_stock_list(self) -> List[str]:
        """获取CSI300成分股完整列表"""
        # 常见的CSI300成分股（部分列表，用于测试）
        return [
            # 银行
            "SH600000", "SH600036", "SH601166", "SZ000001", "SH601318", "SH601288",
            "SH601398", "SH600016", "SH601328", "SH601939", "SH601998", "SH601169",
            
            # 白酒食品
            "SH600519", "SZ000858", "SH603288", "SZ000596", "SH600809", "SZ002304",
            "SZ000799", "SH603369", "SZ002142", "SH600887", "SH603369",
            
            # 医药
            "SH600276", "SZ000661", "SZ002415", "SZ300015", "SH603259", "SZ002007",
            "SZ300347", "SZ300760", "SH600867", "SZ300122", "SZ002821",
            
            # 科技
            "SZ000063", "SZ002415", "SZ300059", "SZ002230", "SZ000725", "SZ300316",
            "SZ002179", "SZ300124", "SZ300182", "SZ002153", "SZ300408",
            
            # 房地产
            "SZ000002", "SZ000001", "SH600340", "SH601155", "SZ000069", "SH600048",
            "SZ001979", "SH600606", "SZ000656", "SH600383",
            
            # 能源化工
            "SH600028", "SH600519", "SH601857", "SH600029", "SH600688", "SH600256",
            "SH601012", "SH600740", "SH601808", "SH600009", "SH600309",
            
            # 汽车
            "SH600104", "SZ000625", "SH601238", "SZ002594", "SH601633", "SZ000800",
            "SZ000825", "SH600741", "SZ002460", "SZ300750",
            
            # 电子
            "SZ002415", "SZ000725", "SZ002230", "SZ300059", "SZ002179", "SZ300124",
            "SZ002153", "SZ300316", "SZ000063", "SZ300182",
            
            # 家电
            "SZ000333", "SZ000651", "SZ002415", "SH600690", "SZ000651", "SZ002032",
            "SZ000921", "SZ000100", "SH603833", "SZ002050",
            
            # 基建
            "SH601668", "SH600015", "SH601186", "SH601390", "SH600795", "SH601618",
            "SH600548", "SH601117", "SH601800", "SH600970",
            
            # 钢铁有色
            "SH600019", "SH601899", "SZ000776", "SH601600", "SH600362", "SH603993",
            "SH601088", "SZ000878", "SH600111", "SH601158",
            
            # 其他重要股票
            "SH600585", "SH601012", "SH601919", "SH600036", "SH600009", "SH600030",
            "SH600196", "SH600332", "SH600362", "SH600406", "SH600438", "SH600456",
            "SH600498", "SH600519", "SH600585", "SH600663", "SH600690", "SH600795",
            "SH600816", "SH600837", "SH600887", "SH600919", "SH600958", "SH600999",
            "SH601012", "SH601088", "SH601117", "SH601186", "SH601238", "SH601288",
            "SH601318", "SH601328", "SH601390", "SH601398", "SH601618", "SH601633",
            "SH601668", "SH601800", "SH601857", "SH601899", "SH601919", "SH601939",
            "SH601998", "SH603288", "SH603369", "SH603833", "SH603993",
            "SZ000001", "SZ000002", "SZ000063", "SZ000069", "SZ000100", "SZ000333",
            "SZ000596", "SZ000625", "SZ000651", "SZ000656", "SZ000661", "SZ000725",
            "SZ000776", "SZ000799", "SZ000800", "SZ000825", "SZ000858", "SZ000876",
            "SZ000878", "SZ000921", "SZ001979", "SZ002007", "SZ002032", "SZ002050",
            "SZ002142", "SZ002153", "SZ002179", "SZ002230", "SZ002304", "SZ002415",
            "SZ002460", "SZ002594", "SZ002821", "SZ300015", "SZ300059", "SZ300122",
            "SZ300124", "SZ300182", "SZ300316", "SZ300347", "SZ300408", "SZ300750",
            "SZ300760"
        ]

    def measure_memory_usage(self) -> float:
        """测量当前内存使用量 (MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def test_basic_features(self, 
                          instruments: List[str],
                          start_time: str = "2020-01-01",
                          end_time: str = "2020-03-01") -> List[Dict]:
        """测试基础特征性能"""
        
        basic_features = [
            ("$close", "收盘价"),
            ("$volume", "成交量"),
            ("$open", "开盘价"),
            ("$high", "最高价"),
            ("$low", "最低价"),
            ("$vwap", "成交量加权平均价"),
            ("Mean($close, 5)", "5日均价"),
            ("Mean($close, 20)", "20日均价"),
            ("Std($close, 20)", "20日收盘价标准差"),
            ("($close / Ref($close, 1)) - 1", "日收益率"),
        ]
        
        print(f"🔍 测试基础特征性能")
        print(f"📊 股票数量: {len(instruments)}")
        print(f"📅 时间范围: {start_time} 到 {end_time}")
        print("=" * 80)
        
        results = []
        
        for i, (feature_expr, feature_name) in enumerate(basic_features):
            print(f"\n[{i+1:2d}/{len(basic_features)}] 测试: {feature_name}")
            print(f"   表达式: {feature_expr}")
            
            result = self._test_single_expression(
                expression=feature_expr,
                name=feature_name,
                instruments=instruments,
                start_time=start_time,
                end_time=end_time
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def test_alpha_expressions(self,
                             instruments: List[str],
                             start_time: str = "2020-01-01", 
                             end_time: str = "2020-03-01") -> List[Dict]:
        """测试Alpha表达式性能"""
        
        alpha_features = [
            ("(-1 * Corr(CSRank(Delta(Log($volume), 1)), CSRank((($close - $open) / $open)), 6))", "ALPHA1"),
            ("(-1 * Delta((($close - $low) - ($high - $close)) / ($high - $low), 1))", "ALPHA2"),
            ("Sum(If($close == Ref($close,1), 0, $close - If($close > Ref($close,1), Less($low, Ref($close,1)), Greater($high, Ref($close,1)))), 6)", "ALPHA3"),
            ("If((Sum($close,8)/8 + Std($close,8)) < (Sum($close,2)/2), -1, If((Sum($close,2)/2) < (Sum($close,8)/8 - Std($close,8)), 1, If(($volume / Mean($volume,20)) >= 1, 1, -1)))", "ALPHA4"),
            ("-1 * Corr(TSRank($volume,5), TSRank($high,5),5)", "ALPHA5"),
            ("(CSRank(Sign(Delta((($open * 0.85) + ($high * 0.15)), 4))) * -1)", "ALPHA6"),
            ("(CSRank(Greater(($vwap - $close), 3)) + CSRank(Min(($vwap - $close), 3))) * CSRank(Delta($volume, 3))", "ALPHA7"),
            ("CSRank(Delta((((($high + $low)/2) * 0.2) + ($vwap * 0.8)), 4) * -1)", "ALPHA8"),
            ("EMA((((($high + $low)/2) - ((Ref($high,1) + Ref($low,1))/2)) * (($high - $low)/$volume)), 4)", "ALPHA9"),
            ("CSRank(Greater(Power(If(($close / Ref($close, 1)) - 1 < 0, Std(($close / Ref($close, 1)) - 1, 20), $close), 2), 5))", "ALPHA10"),
        ]
        
        print(f"\n🚀 测试Alpha表达式性能")
        print(f"📊 股票数量: {len(instruments)}")
        print(f"📅 时间范围: {start_time} 到 {end_time}")
        print("=" * 80)
        
        results = []
        
        for i, (feature_expr, feature_name) in enumerate(alpha_features):
            print(f"\n[{i+1:2d}/{len(alpha_features)}] 测试: {feature_name}")
            print(f"   表达式: {feature_expr}")
            
            result = self._test_single_expression(
                expression=feature_expr,
                name=feature_name,
                instruments=instruments,
                start_time=start_time,
                end_time=end_time
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def _test_single_expression(self,
                              expression: str,
                              name: str,
                              instruments: List[str],
                              start_time: str,
                              end_time: str,
                              timeout: float = 120.0) -> Dict:
        """测试单个表达式"""
        
        result = {
            'name': name,
            'expression': expression,
            'status': 'unknown',
            'error': None,
            'execution_time': None,
            'memory_before': None,
            'memory_after': None,
            'memory_usage': None,
            'data_shape': None,
            'instruments_count': len(instruments),
            'date_range': f"{start_time} to {end_time}",
            'data_quality': None,
            'warnings': []
        }
        
        try:
            # 记录初始内存
            gc.collect()
            result['memory_before'] = self.measure_memory_usage()
            
            # 捕获警告
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # 执行表达式计算
                start_exec = time.time()
                
                df = D.features(
                    instruments,
                    [expression],
                    start_time=start_time,
                    end_time=end_time
                )
                
                end_exec = time.time()
                
                # 记录警告
                result['warnings'] = [str(warning.message) for warning in w]
            
            # 记录执行时间
            result['execution_time'] = end_exec - start_exec
            
            # 记录数据信息
            result['data_shape'] = df.shape
            
            # 数据质量分析
            if not df.empty:
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    result['data_quality'] = {
                        'total_values': df.size,
                        'nan_count': df.isnull().sum().sum(),
                        'nan_ratio': df.isnull().sum().sum() / df.size,
                        'inf_count': np.isinf(numeric_df).sum().sum(),
                        'zero_count': (numeric_df == 0).sum().sum(),
                        'unique_instruments': df.index.get_level_values('instrument').nunique(),
                        'date_count': df.index.get_level_values('datetime').nunique(),
                        'value_stats': {
                            'min': float(numeric_df.min().min()),
                            'max': float(numeric_df.max().max()), 
                            'mean': float(numeric_df.mean().mean()),
                            'std': float(numeric_df.std().mean()),
                        }
                    }
            
            # 记录结束内存
            result['memory_after'] = self.measure_memory_usage()
            result['memory_usage'] = result['memory_after'] - result['memory_before']
            
            result['status'] = 'success'
            
            print(f"   ✅ 成功! 耗时: {result['execution_time']:.3f}s, 数据: {result['data_shape']}, 内存: +{result['memory_usage']:.1f}MB")
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['memory_after'] = self.measure_memory_usage()
            if result['memory_before']:
                result['memory_usage'] = result['memory_after'] - result['memory_before']
            
            print(f"   ❌ 失败: {result['error']}")
            
            # 错误分类
            if "window must be an integer" in str(e):
                result['error_type'] = 'window_parameter_error'
            elif "unsupported operand type" in str(e):
                result['error_type'] = 'operator_error'
            elif "not found" in str(e).lower():
                result['error_type'] = 'data_not_found'
            else:
                result['error_type'] = 'unknown_error'
        
        return result
    
    def test_scalability(self,
                        base_instruments: List[str],
                        test_sizes: List[int] = None,
                        start_time: str = "2020-01-01",
                        end_time: str = "2020-02-01") -> List[Dict]:
        """测试可扩展性性能"""
        
        if test_sizes is None:
            test_sizes = [10, 30, 50, 100, len(base_instruments)]
        
        # 确保测试大小不超过可用股票数量
        test_sizes = [size for size in test_sizes if size <= len(base_instruments)]
        
        test_expression = "Mean($close, 5)"
        test_name = "可扩展性测试"
        
        print(f"\n📈 测试可扩展性性能")
        print(f"🧪 测试表达式: {test_expression}")
        print(f"📊 测试规模: {test_sizes}")
        print("=" * 80)
        
        results = []
        
        for size in test_sizes:
            instruments = base_instruments[:size]
            print(f"\n🔍 测试股票数量: {size}")
            
            result = self._test_single_expression(
                expression=test_expression,
                name=f"{test_name}_{size}股票",
                instruments=instruments,
                start_time=start_time,
                end_time=end_time
            )
            
            result['test_size'] = size
            results.append(result)
            self.results.append(result)
        
        return results
    
    def generate_csi300_report(self, results: List[Dict] = None) -> str:
        """生成CSI300专用性能报告"""
        
        results = results or self.results
        
        if not results:
            return "没有测试结果"
        
        # 分类结果
        basic_results = [r for r in results if '测试' not in r['name'] and 'ALPHA' not in r['name']]
        alpha_results = [r for r in results if 'ALPHA' in r['name']]
        scalability_results = [r for r in results if '测试' in r['name']]
        
        # 统计信息
        total_tests = len(results)
        successful_tests = len([r for r in results if r['status'] == 'success'])
        failed_tests = total_tests - successful_tests
        
        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("📊 CSI300 数据性能测试报告")
        report.append("=" * 80)
        
        # 总体统计
        report.append(f"\n📈 总体统计:")
        report.append(f"   总测试数量: {total_tests}")
        report.append(f"   成功数量: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        report.append(f"   失败数量: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        
        # 性能分析
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            execution_times = [r['execution_time'] for r in successful_results]
            memory_usage = [r['memory_usage'] for r in successful_results if r['memory_usage'] is not None]
            
            report.append(f"\n⏱️  执行时间分析:")
            report.append(f"   平均执行时间: {np.mean(execution_times):.3f}s")
            report.append(f"   最快: {np.min(execution_times):.3f}s")
            report.append(f"   最慢: {np.max(execution_times):.3f}s")
            report.append(f"   中位数: {np.median(execution_times):.3f}s")
            
            if memory_usage:
                report.append(f"\n💾 内存使用分析:")
                report.append(f"   平均内存增长: {np.mean(memory_usage):.2f} MB")
                report.append(f"   最大内存增长: {np.max(memory_usage):.2f} MB")
                report.append(f"   最小内存增长: {np.min(memory_usage):.2f} MB")
        
        # 基础特征性能
        if basic_results:
            report.append(f"\n🔍 基础特征性能:")
            report.append("-" * 60)
            for result in basic_results:
                if result['status'] == 'success':
                    report.append(f"   ✅ {result['name']:15s}: {result['execution_time']:.3f}s, {result['data_shape']}")
                else:
                    report.append(f"   ❌ {result['name']:15s}: {result['error']}")
        
        # Alpha表达式性能
        if alpha_results:
            report.append(f"\n🚀 Alpha表达式性能:")
            report.append("-" * 60)
            for result in alpha_results:
                if result['status'] == 'success':
                    report.append(f"   ✅ {result['name']:20s}: {result['execution_time']:.3f}s, {result['data_shape']}")
                else:
                    report.append(f"   ❌ {result['name']:20s}: {result['error']}")
        
        # 可扩展性分析
        if scalability_results:
            report.append(f"\n📈 可扩展性分析:")
            report.append("-" * 60)
            
            success_scalability = [r for r in scalability_results if r['status'] == 'success']
            if len(success_scalability) > 1:
                sizes = [r['test_size'] for r in success_scalability]
                times = [r['execution_time'] for r in success_scalability]
                
                # 计算时间复杂度
                if len(sizes) >= 2:
                    time_growth_ratio = times[-1] / times[0]
                    size_growth_ratio = sizes[-1] / sizes[0]
                    complexity_estimate = np.log(time_growth_ratio) / np.log(size_growth_ratio)
                    
                    report.append(f"   股票数量范围: {min(sizes)} - {max(sizes)}")
                    report.append(f"   时间增长倍数: {time_growth_ratio:.2f}x")
                    report.append(f"   估计时间复杂度: O(n^{complexity_estimate:.2f})")
            
            for result in success_scalability:
                report.append(f"   {result['test_size']:3d} 股票: {result['execution_time']:.3f}s")
        
        # 数据质量分析
        quality_results = [r for r in successful_results if r['data_quality']]
        if quality_results:
            report.append(f"\n📊 数据质量分析:")
            report.append("-" * 60)
            
            total_nan_ratio = np.mean([r['data_quality']['nan_ratio'] for r in quality_results])
            avg_instruments = np.mean([r['data_quality']['unique_instruments'] for r in quality_results])
            avg_dates = np.mean([r['data_quality']['date_count'] for r in quality_results])
            
            report.append(f"   平均缺失值比例: {total_nan_ratio:.3f}")
            report.append(f"   平均股票数量: {avg_instruments:.0f}")
            report.append(f"   平均交易日数: {avg_dates:.0f}")
        
        # 优化建议
        report.append(f"\n💡 优化建议:")
        
        if failed_tests > 0:
            report.append(f"   🔧 修复 {failed_tests} 个失败的表达式")
        
        slow_tests = [r for r in successful_results if r['execution_time'] > 2.0]
        if slow_tests:
            report.append(f"   ⚡ 优化 {len(slow_tests)} 个执行时间超过2秒的表达式")
        
        high_memory_tests = [r for r in successful_results if r.get('memory_usage', 0) > 200]
        if high_memory_tests:
            report.append(f"   💾 优化 {len(high_memory_tests)} 个内存使用超过200MB的表达式")
        
        # CSI300特定建议
        report.append(f"\n📈 CSI300特定建议:")
        report.append(f"   📊 CSI300包含300只活跃股票，建议使用批量计算优化")
        report.append(f"   🕐 对于日频数据，建议缓存中间计算结果")
        report.append(f"   💻 对于复杂Alpha表达式，考虑并行计算")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "csi300_performance_report.txt"):
        """保存结果"""
        import json
        
        # 保存详细结果
        json_filename = filename.replace('.txt', '.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存报告
        report = self.generate_csi300_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 CSI300测试结果已保存:")
        print(f"   详细结果: {json_filename}")
        print(f"   性能报告: {filename}")

def main():
    """主函数"""
    
    print("🚀 CSI300 数据性能测试")
    print("=" * 80)
    
    # 创建测试器
    tester = CSI300PerformanceTester()
    
    # 获取CSI300股票池
    print("📊 获取CSI300成分股...")
    instruments = tester.get_csi300_instruments(sample_size=200)  # 采样30只股票进行测试
    print(f"✅ 获取到 {len(instruments)} 只股票")
    
    # 测试配置
    start_time = "2008-01-01"
    end_time = "2025-07-15"  # 2个月的数据用于快速测试
    
    try:
        # 1. 测试基础特征
        print(f"\n🔍 阶段1: 基础特征性能测试")
        #basic_results = tester.test_basic_features(instruments, start_time, end_time)
        
        # 2. 测试Alpha表达式
        print(f"\n🚀 阶段2: Alpha表达式性能测试")
        alpha_results = tester.test_alpha_expressions(instruments, start_time, end_time)
        
        # 3. 测试可扩展性
        print(f"\n📈 阶段3: 可扩展性性能测试")
        # scalability_results = tester.test_scalability(
        #     instruments,
        #     test_sizes=[10, 20, 30, min(50, len(instruments))],
        #     start_time=start_time,
        #     end_time="2020-01-15"  # 缩短时间用于可扩展性测试
        # )
        
        # 生成和显示报告
        print(f"\n📊 生成性能报告...")
        report = tester.generate_csi300_report()
        print("\n" + report)
        
        # 保存结果
        tester.save_results()
        
        print(f"\n🎉 CSI300性能测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 