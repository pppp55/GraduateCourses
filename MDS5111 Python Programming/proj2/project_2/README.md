# Environment Setup and Running Guide

You should only implement your codes in the `./src/project.py`. When grading, we will
only copy your `./src/project.py` file to our folder (containing orginal `config.py` and `utils.py`)
and execute it.

## Grading environment
You are recommended to use the official grading environment's Python and package 
versions in this repository. The grading environment uses the same setup. 
Running your code in a different environment may lead to installation issues or output differences.

The official Python runtime exported from the grading environment is **Python 3.10.20**.
If you want to use the same environment, we suggest that please do not modify your system 
Python installation for this assignment. Instead, we strongly recommend using `conda` to 
create a new environment with Python 3.10.20.

You are recommended use the following one-command setup to keep your environment
exactly same as the grading environment.

### Windows

```powershell
py -3.10 setup_and_run.py
```

or:

```powershell
.\setup_and_run.ps1
```

### macOS / Linux

```bash
python3.10 setup_and_run.py
```

or:

```bash
bash setup_and_run.sh
```

The above scripts will:
1. create a local virtual environment named `.venv`
2. upgrade `pip`
3. install all packages from `requirements.txt`
4. set the official runtime environment variables
5. run:
```bash
python src/project.py --data_dir data --output_dir outputs --figure_dir figures
```