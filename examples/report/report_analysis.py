import multiprocessing  # 用于Windows多进程修复
import argparse
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.contrib.report import analysis_position, analysis_model
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.stats import pearsonr, spearmanr
from datetime import datetime
import calendar  # For month names
from qlib.data import D

# 常量定义：颜色方案和布局设置
COLORS = {
    'strategy': '#1f77b4',
    'benchmark': '#ff7f0e',
    'excess': '#2ca02c',
    'drawdown': '#d62728',
    'turnover': '#9467bd',
    'ic': '#1f77b4',
    'rank_ic': '#ff7f0e'
}
LAYOUT_HEIGHT_PER_SUBPLOT = 300
LEGEND_CONFIG_BASE = dict(
    orientation="v",
    x=1.02,
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor="rgba(0,0,0,0.3)",
    borderwidth=1,
    font=dict(size=10)
)


def add_cumulative_returns(fig, report_df, row, col, legend_group):
    """添加累积收益对比曲线到子图"""
    if 'return' in report_df.columns:
        cumulative_return = (1 + report_df['return']).cumprod() - 1
        fig.add_trace(go.Scatter(x=report_df.index, y=cumulative_return, mode='lines', name='策略累积收益',
                                 line=dict(color=COLORS['strategy'], width=2), legend=legend_group, showlegend=True),
                      row=row, col=col)
    if 'bench' in report_df.columns:
        benchmark_return = (1 + report_df['bench']).cumprod() - 1
        fig.add_trace(go.Scatter(x=report_df.index, y=benchmark_return, mode='lines', name='基准累积收益',
                                 line=dict(color=COLORS['benchmark'], width=2), legend=legend_group, showlegend=True),
                      row=row, col=col)
    if 'return' in report_df.columns and 'bench' in report_df.columns:
        excess_return = cumulative_return - benchmark_return
        fig.add_trace(go.Scatter(x=report_df.index, y=excess_return, mode='lines', name='超额收益',
                                 line=dict(color=COLORS['excess'], width=2), legend=legend_group, showlegend=True),
                      row=row, col=col)

def add_return_distribution(fig, report_df, row, col, legend_group):
    """添加每日收益分布对比到子图"""
    if 'return' in report_df.columns:
        fig.add_trace(go.Histogram(x=report_df['return'], nbinsx=50, name='策略收益分布',
                                   marker_color=COLORS['strategy'], opacity=0.7, legend=legend_group, showlegend=True),
                      row=row, col=col)
    if 'bench' in report_df.columns:
        fig.add_trace(go.Histogram(x=report_df['bench'], nbinsx=50, name='基准收益分布',
                                   marker_color=COLORS['benchmark'], opacity=0.7, legend=legend_group, showlegend=True),
                      row=row, col=col)

def add_drawdown(fig, report_df, row, col, legend_group):
    """添加回撤分析到子图"""
    if 'return' in report_df.columns:
        cumulative_return = (1 + report_df['return']).cumprod()
        running_max = cumulative_return.cummax()
        drawdown = (cumulative_return - running_max) / running_max
        fig.add_trace(go.Scatter(x=report_df.index, y=drawdown, mode='lines', name='策略回撤',
                                 line=dict(color=COLORS['drawdown']), fill='tonexty', legend=legend_group, showlegend=True),
                      row=row, col=col)
    if 'bench' in report_df.columns:
        benchmark_cumulative = (1 + report_df['bench']).cumprod()
        benchmark_running_max = benchmark_cumulative.cummax()
        benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
        fig.add_trace(go.Scatter(x=report_df.index, y=benchmark_drawdown, mode='lines', name='基准回撤',
                                 line=dict(color=COLORS['benchmark'], dash='dash'), legend=legend_group, showlegend=True),
                      row=row, col=col)

def add_turnover(fig, report_df, row, col, legend_group):
    """添加换手率到子图"""
    turnover_col = 'turnover' if 'turnover' in report_df.columns else 'turn' if 'turn' in report_df.columns else None
    if turnover_col:
        fig.add_trace(go.Scatter(x=report_df.index, y=report_df[turnover_col], mode='lines', name='换手率',
                                 line=dict(color=COLORS['turnover'], width=2), legend=legend_group, showlegend=True),
                      row=row, col=col)

def add_performance_metrics(fig, report_df, row, col):
    """添加关键指标文本总结到子图"""
    metrics_lines = []
    if 'return' in report_df.columns:
        returns = report_df['return']
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        cumulative_return = (1 + returns).cumprod()
        drawdown = (cumulative_return - cumulative_return.cummax()) / cumulative_return.cummax()
        max_drawdown = drawdown.min() if not drawdown.empty else 0

        metrics_lines.extend([
            f"策略年化收益: {annual_return:.2%}",
            f"策略年化波动: {volatility:.2%}",
            f"策略夏普比率: {sharpe_ratio:.3f}",
            f"策略最大回撤: {max_drawdown:.2%}"
        ])

        if 'bench' in report_df.columns:
            bench_returns = report_df['bench']
            bench_total_return = (1 + bench_returns).prod() - 1
            bench_annual_return = (1 + bench_total_return) ** (252 / len(bench_returns)) - 1 if len(bench_returns) > 0 else 0
            bench_volatility = bench_returns.std() * np.sqrt(252) if len(bench_returns) > 1 else 0
            bench_sharpe = bench_annual_return / bench_volatility if bench_volatility > 0 else 0
            bench_cumulative = (1 + bench_returns).cumprod()
            bench_drawdown = (bench_cumulative - bench_cumulative.cummax()) / bench_cumulative.cummax()
            bench_max_drawdown = bench_drawdown.min() if not bench_drawdown.empty else 0

            excess_annual = annual_return - bench_annual_return
            tracking_error = (returns - bench_returns).std() * np.sqrt(252) if len(returns) > 1 else 0
            information_ratio = excess_annual / tracking_error if tracking_error > 0 else 0

            metrics_lines.extend([
                f"基准年化收益: {bench_annual_return:.2%}",
                f"超额年化收益: {excess_annual:.2%}",
                f"信息比率: {information_ratio:.3f}"
            ])

    turnover_col = 'turnover' if 'turnover' in report_df.columns else 'turn' if 'turn' in report_df.columns else None
    if turnover_col:
        avg_turnover = report_df[turnover_col].mean()
        metrics_lines.append(f"平均换手率: {avg_turnover:.2%}")

    fig.add_annotation(
        text="<br>".join(metrics_lines),
        xref=f"x{row}", yref=f"y{row}",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="black"),
        align="left",
        bordercolor="black",
        borderwidth=2,
        bgcolor="rgba(255,255,255,0.9)",
        xanchor="center",
        yanchor="middle"
    )

def create_position_analysis_plots(report_df):
    """使用 Plotly 创建投资组合分析图表（单列布局）"""
    if report_df is None or report_df.empty:
        print("⚠️ 投资组合报告数据为空")
        return None

    print(f"📊 投资组合数据列: {report_df.columns.tolist()}")

    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=[
            '累积收益对比 (Cumulative Returns)',
            '每日收益分布 (Daily Returns Distribution)',
            '最大回撤 (Maximum Drawdown)',
            '换手率 (Turnover)',
            '收益指标总结 (Performance Metrics)'
        ],
        vertical_spacing=0.08
    )

    # 为每个subplot定义独立的legend
    legend_configs = {}
    num_rows = 5
    for row in range(1, num_rows + 1):
        legend_name = f'legend{row}'
        y_pos = 1 - (row - 1) / num_rows - 0.5 / num_rows  # 居中于subplot
        legend_configs[legend_name] = {**LEGEND_CONFIG_BASE, 'y': y_pos, 'yanchor': 'middle'}

    add_cumulative_returns(fig, report_df, row=1, col=1, legend_group='legend1')
    add_return_distribution(fig, report_df, row=2, col=1, legend_group='legend2')
    add_drawdown(fig, report_df, row=3, col=1, legend_group='legend3')
    add_turnover(fig, report_df, row=4, col=1, legend_group='legend4')
    add_performance_metrics(fig, report_df, row=5, col=1)

    fig.update_layout(
        title_text="📈 投资组合综合分析报告",
        title_x=0.5,
        height=LAYOUT_HEIGHT_PER_SUBPLOT * 5,
        showlegend=True,
        **legend_configs  # 添加所有legend配置
    )

    # 更新轴标签
    fig.update_xaxes(title_text="日期", row=1, col=1)
    fig.update_yaxes(title_text="累积收益率", row=1, col=1)
    fig.update_xaxes(title_text="日收益率", row=2, col=1)
    fig.update_yaxes(title_text="频次", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=3, col=1)
    fig.update_yaxes(title_text="回撤幅度", row=3, col=1)
    fig.update_xaxes(title_text="日期", row=4, col=1)
    fig.update_yaxes(title_text="换手率", row=4, col=1)
    fig.update_xaxes(visible=False, row=5, col=1)
    fig.update_yaxes(visible=False, row=5, col=1)

    return fig

def combine_model_figures(model_figures):
    """将多个模型分析图表合并到一个子图中（每行一个图表）"""
    if not isinstance(model_figures, list) or not model_figures:
        print("⚠️ 无模型图表可合并")
        return None

    num_figures = len(model_figures)
    subplot_titles = [
                         "累积收益 (Cumulative Return)",
                         "收益分布 (Return Distribution)",
                         "信息系数 (Information Coefficient)",
                         "月度IC热力图 (Monthly IC Heatmap)",
                         "IC分布直方图 (IC Distribution Histogram)",
                         "IC正态分布Q-Q图 (IC Normal Distribution Q-Q Plot)",
                         "自相关性 (Auto Correlation)"
                     ][:num_figures + 1]  # 调整标题数量，考虑IC分布拆分

    fig_combined = make_subplots(
        rows=num_figures + 1, cols=1,  # +1 for potential split IC dist
        subplot_titles=subplot_titles,
        vertical_spacing=0.05
    )

    # 为每个subplot定义独立的legend
    legend_configs = {}
    effective_rows = num_figures + 1  # 包括潜在拆分
    current_row = 1
    for row in range(1, effective_rows + 1):
        legend_name = f'legend{row}'
        y_pos = 1 - (row - 1) / effective_rows - 0.5 / effective_rows  # 居中于subplot
        legend_configs[legend_name] = {**LEGEND_CONFIG_BASE, 'y': y_pos, 'yanchor': 'middle'}

    for i, fig in enumerate(model_figures):
        if fig is None or not hasattr(fig, 'data'):
            continue

        legend_group = f'legend{current_row}'

        if i == 4:  # IC Distribution图特殊处理
            histogram_traces = [t for t in fig.data if 'x2' not in str(t.xaxis)]
            qq_traces = [t for t in fig.data if 'x2' in str(t.xaxis)]
            for trace in histogram_traces:
                trace.legend = legend_group
                trace.showlegend = True  # 假设需要显示legend，如果有
                fig_combined.add_trace(trace, row=current_row, col=1)
            current_row += 1
            legend_group = f'legend{current_row}'  # 更新为下一个row的legend
            for trace in qq_traces:
                trace.legend = legend_group
                trace.showlegend = True
                fig_combined.add_trace(trace, row=current_row, col=1)
            current_row += 1
            continue

        for trace in fig.data:
            trace.legend = legend_group
            trace.showlegend = (i in [0, 2])  # 为累积收益 (i=0) 和IC图 (i=2) 启用legend
            if i == 3 and trace.type == 'heatmap':  # 热力图colorbar动态位置
                trace.colorbar = dict(
                    x=1.02,
                    y=1 - (current_row - 1) / effective_rows - 0.5 / effective_rows,
                    len=1 / effective_rows,
                    thickness=15,
                    title={'text': 'IC值', 'side': 'right'}
                )
            fig_combined.add_trace(trace, row=current_row, col=1)

        current_row += 1

    fig_combined.update_layout(
        title_text="📊 模型性能综合分析报告",
        title_x=0.5,
        height=LAYOUT_HEIGHT_PER_SUBPLOT * effective_rows,
        showlegend=True,
        **legend_configs  # 添加所有legend配置
    )

    # 更新轴标签（动态）
    axis_updates = [
        {"x": "日期", "y": "累积收益"},
        {"x": "收益率", "y": "密度"},
        {"x": "日期", "y": "IC值"},
        {"x": "月份", "y": "年份"},
        {"x": "IC值", "y": "概率密度"},
        {"x": "理论分位数", "y": "观测分位数"},
        {"x": "日期", "y": "自相关系数"}
    ]
    for row, updates in enumerate(axis_updates[:current_row - 1], 1):
        fig_combined.update_xaxes(title_text=updates["x"], row=row, col=1)
        fig_combined.update_yaxes(title_text=updates["y"], row=row, col=1)

    return fig_combined

def create_portfolio_calendar(recorder):
    """创建投资组合日历热力图（按年分子图，x轴为每月日期，y轴为月份）"""
    try:
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    except Exception as e:
        print(f"⚠️ 加载持仓数据失败: {e}")
        return None

    # 假设positions是dict: date -> Position object
    dates = sorted(positions.keys())
    calendar_data = []
    prev_pos = None
    for idx, date in enumerate(dates):
        pos = positions[date]
        cash = pos.get_cash()
        total_value = pos.calculate_value() if hasattr(pos, 'calculate_value') else 1  # 防止除零
        turnover = 0.0
        trades_info = []
        if prev_pos is not None and hasattr(pos, 'position') and hasattr(prev_pos, 'position'):
            current_pos = pos.position
            prev_pos_dict = prev_pos.position
            all_instr = set(current_pos.keys()).union(prev_pos_dict.keys())
            buy_value = sell_value = 0
            for inst in all_instr:
                if inst in {'cash', 'now_account_value'}:
                    continue
                curr_amount = current_pos.get(inst, {}).get('amount', 0)
                prev_amount = prev_pos_dict.get(inst, {}).get('amount', 0)
                curr_price = current_pos.get(inst, {}).get('price', 0)
                prev_price = prev_pos_dict.get(inst, {}).get('price', 0)
                delta = curr_amount - prev_amount
                if delta > 0:
                    buy_value += delta * curr_price
                    trades_info.append(f"Bought {inst}: {delta:.0f} shares")
                elif delta < 0:
                    sell_value += -delta * prev_price
                    trades_info.append(f"Sold {inst}: {-delta:.0f} shares")
            turnover = (buy_value + sell_value) / total_value if total_value > 0 else 0
        trades_str = '<br>'.join(trades_info) if trades_info else 'No Trades'
        position = 100 * (total_value - cash) / total_value if total_value > 0 else 0
        calendar_data.append({'date': date, 'turnover': turnover, 'position': position, 'trades': trades_str})
        prev_pos = pos

    if not calendar_data:
        print("⚠️ 无持仓数据可用于生成日历")
        return None

    df = pd.DataFrame(calendar_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 为了显示完整日历，包括周末（但无数据，显示为空）
    df = df.resample('D').ffill()  # 填充到每天，但turnover只在交易日有值，周末会ffill，但我们将在pivot中处理
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    df['weekday'] = df.index.weekday  # 0=Mon to 6=Sun
    # 对于非交易日（假设周末无交易），设置turnover为nan
    df.loc[df['weekday'] >= 5, 'turnover'] = np.nan
    df.loc[df['weekday'] >= 5, 'trades'] = 'No Data'

    # 按年分组创建子图
    years = sorted(df['year'].unique())
    num_years = len(years)
    if num_years == 0:
        print("⚠️ 无有效年份数据")
        return None

    fig = make_subplots(rows=num_years, cols=1, subplot_titles=[f"Year {year}" for year in years], vertical_spacing=0.05, shared_xaxes=False)

    max_turnover = df['turnover'].max() * 100 if not df['turnover'].empty else 25

    for i, year in enumerate(years, 1):
        year_df = df[df['year'] == year]
        pivot_data = pd.pivot_table(year_df, values='turnover', index='month', columns='day_of_month', aggfunc='first')  # first 因为ffill
        pivot_hover = pd.pivot_table(year_df, values='trades', index='month', columns='day_of_month', aggfunc='first')
        pivot_value = pd.pivot_table(year_df, values='position', index='month', columns='day_of_month', aggfunc='first')

        # Reindex to ensure all 12 months and 31 days
        pivot_data = pivot_data.reindex(index=range(1,13), columns=range(1,32), fill_value=0)
        pivot_hover = pivot_hover.reindex(index=range(1,13), columns=range(1,32), fill_value='No Data')
        pivot_value = pivot_value.reindex(index=range(1,13), columns=range(1,32), fill_value=0)

        z_values = pivot_data.values * 100

        # 处理文本
        text_values = np.vectorize(lambda x: '' if np.isnan(x) or x == 0 else f"{int(round(x))}%" )(z_values)

        # hovertext
        hover_text = np.full_like(z_values, 'No Data', dtype=object)
        for m in range(12):
            for d in range(31):
                month = m + 1
                day = d + 1
                value = z_values[m, d]
                trades = pivot_hover.iloc[m, d]
                total_value = pivot_value.iloc[m,d]
                if value > 0:
                    hover_text[m, d] = f"Date: {year}-{month:02d}-{day:02d}<br>Turnover: {value:.2f}%<br>Position: {total_value:,.2f}<br>{trades}"

        # 添加热力图
        heatmap_trace = go.Heatmap(
            z=z_values,
            x=list(range(1, 32)),
            y=[calendar.month_abbr[m] for m in range(1, 13)],
            colorscale='YlOrRd',
            hoverongaps=False,
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            text=text_values,
            texttemplate="%{text}",
            textfont=dict(size=8, color='black'),
            showscale=True,
            zmin=0,
            zmax=max_turnover,
            connectgaps=False
        )
        fig.add_trace(heatmap_trace, row=i, col=1)

        # Dynamically set colorbar height to match the subplot height
        subplot_height = 600 / num_years  # Total height divided by number of years
        colorbar_y = 1 - (i - 0.5) / num_years  # Center the colorbar vertically in the subplot
        heatmap_trace.colorbar = dict(
            title="换手率 (%)",
            thickness=15,
            len=subplot_height / 600,  # Normalize length to the total figure height
            yanchor="middle",
            y=colorbar_y,
            x=1.02,
            title_side="right"
        )

        fig.update_xaxes(title_text="日期 (日)", row=i, col=1, tickfont=dict(size=9), dtick=1, tickvals=list(range(1, 32)), ticktext=[str(d) for d in range(1, 32)], tickangle=90, showticklabels=True, ticks="outside")
        fig.update_yaxes(title_text="月份", row=i, col=1, autorange="reversed", tickfont=dict(size=12), gridcolor='lightgray')

    fig.update_layout(
        title_text="投资组合日历热力图 (按年分子图)",
        height=600 * num_years,
        width=1200,  # 增加宽度以改善比例
        showlegend=False,
        plot_bgcolor='rgba(240,240,240,0.8)',  # 轻微背景色提升可读性
        margin=dict(l=80, r=80, t=50, b=100),  # 调整边距
    )

    for i in range(1, num_years + 1):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=1, zeroline=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=1, zeroline=False)

    return fig

def create_trades_table(recorder):
    """创建交易记录表格"""
    try:
        trades = recorder.load_object("trades.pkl")  # 假设是DataFrame
    except Exception as e:
        print(f"⚠️ 加载交易数据失败: {e}")
        return None

    if trades.empty:
        print("⚠️ 无交易数据")
        return None

    trades = trades.sort_values('date')  # 按日期排序

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(trades.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[trades[col] for col in trades.columns],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(
        title="交易记录表格 (按日期排序)",
        height=600
    )

    return fig

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows多进程修复

    parser = argparse.ArgumentParser(description="Analyze Qlib experiment recorder.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--rec_id", type=str, required=True, help="Recorder ID")
    args = parser.parse_args()

    qlib.init(provider_uri='data', region=REG_CN)

    try:
        recorder = R.get_recorder(experiment_id=args.exp_name, recorder_id=args.rec_id)
    except ValueError as e:
        print(f"Error: {e}")
        print("Available experiments:")
        for exp_name in R.list_experiments():
            print(exp_name)
        raise

    # 加载数据
    pred_df = recorder.load_object("pred.pkl")
    label_df = recorder.load_object("label.pkl")

    # 打印原始形状以调试
    print(f"pred_df shape: {pred_df.shape}")
    print(f"label_df shape: {label_df.shape}")

    # 合并数据
    pred_label = pd.concat([pred_df, label_df], axis=1, keys=['score', 'label']).reindex(label_df.index)

    # 修复 MultiIndex：扁平化列名
    pred_label.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in pred_label.columns]

    # 确保'score'和'label'列存在
    if 'score' not in pred_label.columns or 'label' not in pred_label.columns:
        pred_label = pred_label.rename(columns={pred_label.columns[0]: 'score', pred_label.columns[-1]: 'label'})

    # 打印修复后形状和列以验证
    print(f"pred_label shape after fix: {pred_label.shape}")
    print(f"pred_label columns: {pred_label.columns.tolist()}")

    # Model analysis
    print("📊 正在生成模型性能分析图表...")
    try:
        model_figures = analysis_model.model_performance_graph(pred_label, show_notebook=False)
        if isinstance(model_figures, list):
            print(f"📈 合并 {len(model_figures)} 个图表到单列子图中...")
            combined_fig = combine_model_figures(model_figures)
            if combined_fig:
                combined_fig.show()
                pio.write_html(combined_fig, file=f'model_analysis_{args.rec_id}.html')
                print(f"📁 模型分析图表已保存: model_analysis_{args.rec_id}.html")
        else:
            model_figures.show()
            pio.write_html(model_figures, file=f'model_analysis_{args.rec_id}.html')
            print(f"📁 模型分析图表已保存: model_analysis_{args.rec_id}.html")
    except Exception as e:
        print(f"⚠️ 模型分析失败: {e}")

    # Position analysis
    print("📊 正在生成投资组合分析图表...")
    try:
        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        position_fig = create_position_analysis_plots(report_df)
        if position_fig:
            position_fig.show()
            pio.write_html(position_fig, file=f'position_analysis_{args.rec_id}.html')
            print(f"📁 投资组合分析图表已保存: position_analysis_{args.rec_id}.html")
    except Exception as e:
        print(f"⚠️ 投资组合分析失败: {e}")

    # 投资组合日历可视化
    print("📅 生成投资组合日历...")
    calendar_fig = create_portfolio_calendar(recorder)
    if calendar_fig:
        calendar_fig.show()
        pio.write_html(calendar_fig, file=f'portfolio_calendar_{args.rec_id}.html')
        print(f"📁 持仓日历已保存: portfolio_calendar_{args.rec_id}.html")
    else:
        print("⚠️ 无法生成持仓日历")

    # 交易记录表格
    print("📋 生成交易记录表格...")
    trades_fig = create_trades_table(recorder)
    if trades_fig:
        trades_fig.show()
        pio.write_html(trades_fig, file=f'trades_table_{args.rec_id}.html')
        print(f"📁 交易表格已保存: trades_table_{args.rec_id}.html")

    print("✅ 所有可视化完成!")