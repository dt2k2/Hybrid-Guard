import sys

from train_url_model import main


if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "svm"])
    if "--svm-kernel" not in sys.argv:
        sys.argv.extend(["--svm-kernel", "linear"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "artifacts/svm_final"])
    main()
