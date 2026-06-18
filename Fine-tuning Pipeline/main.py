"""Pipeline entry point: run data collection game, then fine-tune the model."""
from pathlib import Path

HERE = Path(__file__).parent

if __name__ == "__main__":
    from game import Game
    from finetuning import main as run_finetuning

    csv_path = HERE / "emg_stream.csv"
    model_path = HERE / "final_model.h5"
    output_path = HERE / "finetuned_model.tflite"

    print("=== Starting data collection ===")
    try:
        Game().run()
    except SystemExit:
        pass

    print("\n=== Data collection complete ===")

    if not csv_path.exists():
        print(f"No data file found at {csv_path.name}. Skipping fine-tuning.")
    else:
        print("=== Starting fine-tuning ===\n")
        run_finetuning(
            csv_path=str(csv_path),
            model_path=str(model_path),
            output_path=str(output_path),
        )
        print("\n=== Pipeline complete. Model saved as finetuned_model.tflite ===")
