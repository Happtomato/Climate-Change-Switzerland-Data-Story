import os
import subprocess

def run_notebook():
    subprocess.run(["jupyter", "nbconvert", "--execute", "--to", "notebook", "notebook.ipynb", "--inplace"])

def run_app():
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["run", "notebook", "all"])
    args = parser.parse_args()

    if args.task == "run":
        run_app()
    elif args.task == "notebook":
        run_notebook()
    elif args.task == "all":
        run_notebook()
        run_app()
