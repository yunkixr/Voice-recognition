
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

SCRIPTS = {
    "processing": "main_processing.py",
    "training": "training.py",
    "evaluation": "evaluate.py",
    "converter": "Hzconverter.py",
    "conversation": "conversationprediction.py"
}

ORIGINAL_CONVERSATION_AUDIO = "src/conversation/conversation.wav"
CONVERSATION_AUDIO = "src/conversation/conversation_converted.wav"
MODEL_FILE = "speaker_model.json"


def pause():
    input("\nPress Enter to return to the main menu...")


def find_script(script_name):
    possible_paths = [
        PROJECT_DIR / script_name,
        PROJECT_DIR / "src" / script_name
    ]

    for path in possible_paths:
        if path.is_file():
            return path

    print(f"\nError: Could not find {script_name}.")
    return None


def run_script(script_name, arguments=None):
    script_path = find_script(script_name)

    if script_path is None:
        return False

    module_name = (
        script_path.relative_to(PROJECT_DIR)
        .with_suffix("")
        .as_posix()
        .replace("/", ".")
    )

    command = [
        sys.executable,
        "-m",
        module_name
    ]

    if arguments:
        command.extend(arguments)

    print("\nRunning:", script_name)
    print("-" * 45)

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            check=False
        )

        if result.returncode == 0:
            print("\nOperation completed.")
            return True

        print(
            f"\nOperation failed with exit code "
            f"{result.returncode}."
        )
        return False

    except Exception as error:
        print(f"\nError while running script: {error}")
        return False


def display_csv(file_path):
    path = PROJECT_DIR / file_path

    print(f"\n{file_path}")
    print("-" * 55)

    if not path.is_file():
        print("File not found.")
        return

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as file:
            reader = csv.reader(file)

            for row in reader:
                print(" | ".join(row))

    except Exception as error:
        print(f"Could not read file: {error}")


def display_evaluation(output_directory):
    metrics_file = f"{output_directory}/metrics_summary.csv"
    confusion_file = f"{output_directory}/confusion_matrix.csv"

    print("\nEVALUATION METRICS")
    print("=" * 55)
    display_csv(metrics_file)

    print("\nCONFUSION MATRIX")
    print("=" * 55)
    display_csv(confusion_file)


def extract_features():
    print("\nAUDIO FEATURE EXTRACTION")
    print("=" * 55)

    run_script(SCRIPTS["processing"])
    pause()


def train_model():
    print("\nMODEL TRAINING")
    print("=" * 55)

    confirmation = input(
        "Training will overwrite the existing model and "
        "individual predictions.\n"
        "Do you want to continue? (y/n): "
    ).strip().lower()

    if confirmation == "y":
        run_script(SCRIPTS["training"])
    else:
        print("\nTraining cancelled.")

    pause()


def evaluate_individual_speakers():
    print("\nINDIVIDUAL SPEAKER EVALUATION")
    print("=" * 55)

    run_script(
        SCRIPTS["evaluation"],
        [
            "--predictions",
            "output/predictions.csv",
            "--output-dir",
            "output/evaluation"
        ]
    )

    pause()

def convert_conversation_audio():
    print("\nSAMPLE RATE CONVERSION (44,100 Hz -> 48,000 Hz)")
    print("=" * 55)

    original_path = PROJECT_DIR / ORIGINAL_CONVERSATION_AUDIO

    if not original_path.is_file():
        print(f"Error: Audio file not found: {original_path}")
    else:
        run_script(SCRIPTS["converter"])

    pause()

def predict_conversation():
    print("\nCOMBINED CONVERSATION PREDICTION")
    print("=" * 55)

    original_path = PROJECT_DIR / ORIGINAL_CONVERSATION_AUDIO
    converted_path = PROJECT_DIR / CONVERSATION_AUDIO

    if not (PROJECT_DIR / MODEL_FILE).is_file():
        print(f"Error: Model file not found: {MODEL_FILE}")
        pause()
        return

    # Step 1: convert 44,100 Hz -> 48,000 Hz
    if original_path.is_file():
        print("\nStep 1: Converting audio to 48,000 Hz...")
        converted = run_script(SCRIPTS["converter"])

        if not converted:
            print("Conversion failed. Prediction cancelled.")
            pause()
            return
    elif converted_path.is_file():
        print("\nOriginal recording not found.")
        print("Using the existing converted file.")
    else:
        print(f"Error: Audio file not found: {original_path}")
        pause()
        return

    # Step 2: predict
    print("\nStep 2: Predicting speakers...")
    run_script(
        SCRIPTS["conversation"],
        [
            "--audio",
            CONVERSATION_AUDIO,
            "--model",
            MODEL_FILE,
            "--output",
            "output/conversation_predictions.csv"
        ]
    )

    pause()

def evaluate_conversation():
    print("\nCOMBINED CONVERSATION EVALUATION")
    print("=" * 55)

    predictions_path = (
        PROJECT_DIR / "output/conversation_predictions.csv"
    )

    if not predictions_path.is_file():
        print("Conversation predictions do not exist.")
        print("Run combined conversation prediction first.")
    else:
        run_script(
            SCRIPTS["evaluation"],
            [
                "--predictions",
                "output/conversation_predictions.csv",
                "--output-dir",
                "output/conversation_evaluation"
            ]
        )

    pause()


def view_individual_results():
    print("\nINDIVIDUAL SPEAKER RESULTS")
    print("=" * 55)

    display_evaluation("output/evaluation")
    pause()


def view_conversation_results():
    print("\nCOMBINED CONVERSATION RESULTS")
    print("=" * 55)

    display_evaluation("output/conversation_evaluation")
    pause()


def main_menu():
    while True:
        print("\n")
        print("=" * 55)
        print("       Audio SYSTEM")
        print("=" * 55)

        print("1. Extract Audio Features")
        print("2. Train Audio Model")
        print("3. Evaluate Individual Speakers")
        print("4. Predict Combined Conversation")
        print("5. Evaluate Combined Conversation")
        print("6. View Individual Evaluation Results")
        print("7. View Conversation Evaluation Results")
        print("8. Exit")

        print("=" * 55)

        choice = input("Select an option (1-8): ").strip()

        if choice == "1":
            extract_features()

        elif choice == "2":
            train_model()

        elif choice == "3":
            evaluate_individual_speakers()

        elif choice == "4":
            predict_conversation()

        elif choice == "5":
            evaluate_conversation()

        elif choice == "6":
            view_individual_results()

        elif choice == "7":
            view_conversation_results()

        elif choice == "8":
            print("\nExiting Speaker Classification System.")
            break

        else:
            print("\nInvalid option. Please select 1-8.")
            pause()

if __name__ == "__main__":
    main_menu()
