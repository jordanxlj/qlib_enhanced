import argparse
import qlib
from qlib.workflow import R
from qlib.contrib.report import analysis_position, analysis_model
import pandas as pd

parser = argparse.ArgumentParser(description="Analyze Qlib experiment recorder.")
parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
parser.add_argument("--rec_id", type=str, required=True, help="Recorder ID")
args = parser.parse_args()

qlib.init()

try:
    recorder = R.get_recorder(experiment_id=args.exp_name, recorder_id=args.rec_id)
except ValueError as e:
    print(f"Error: {e}")
    print("Available experiments:")
    for exp_name in R.list_experiments():
        print(exp_name)
    raise

# Model analysis
pred_df = recorder.load_object("pred.pkl")
label_df = recorder.load_object("label.pkl")
pred_label = pd.concat([pred_df, label_df], axis=1, keys=['score', 'label']).reindex(label_df.index)  # Adjust columns to 'score' and 'label'
analysis_model.model_performance_graph(pred_label, show_notebook=True)

# Position analysis
report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
analysis_position.report_graph(report_df, show_notebook=True) 