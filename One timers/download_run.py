import mlflow

mlflow.set_tracking_uri('sqlite:///mlflow.db')
runs = mlflow.search_runs(
    experiment_names=['EMG'],
    filter_string="tags.mlflow.runName = 'EMG-Binary-20260520-144609'"
)

if runs.empty:
    print("Run not found.")
else:
    run_id = runs.iloc[0]['run_id']
    mlflow.artifacts.download_artifacts(run_id=run_id, dst_path='.')
    print(f"Downloaded artifacts for run {run_id}")
