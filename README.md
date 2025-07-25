# AFML - Advances in Financial Machine Learning Solutions

This repository contains solutions to exercises from the book "Advances in Financial Machine Learning" by Marcos Lopez de Prado. Each directory corresponds to a chapter in the book and contains the relevant code and explanations.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project provides a collection of solutions to the exercises in the book "Advances in Financial Machine Learning." The primary goal is to help readers understand and implement the concepts discussed in the book through practical examples. Each chapter's solutions are organized into separate directories for easy navigation. This is intended as a study aid and reference for those working through the book.

## Installation

To quickly get started with these solutions, you can leverage the provided Jupyter notebooks. This ensures all dependencies are correctly installed and the environment is properly configured.

1.  **Prerequisites:**

    *   Python 3.7+
    *   pip
    *   Jupyter Notebook

2.  **Installation Steps:**

    a.  **Clone the repository:**

    bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate.bat # On Windows
        c.  **Install dependencies:**

    bash
    pip install appnope==0.1.4 arch==7.2.0 asttokens==3.0.0 attrs==25.3.0 backtesting==0.6.4 beautifulsoup4==4.13.3 bokeh==3.7.2 certifi==2025.6.15 cffi==1.17.1 charset-normalizer==3.4.1 clarabel==0.11.1 cma==4.2.0 comm==0.2.2 contourpy==1.3.1 curl_cffi==0.11.1 cvxpy==1.6.6 cycler==0.12.1 debugpy==1.8.11 decorator==5.2.1 dill==0.4.0 eodhd==1.0.32 et_xmlfile==2.0.0 exceptiongroup==1.2.2 executing==2.1.0 filelock==3.18.0 finnhub-python==2.4.23 fonttools==4.56.0 frozendict==2.4.6 fsspec==2025.3.2 h11==0.16.0 hmmlearn==0.3.3 idna==3.10 importlib_metadata==8.6.1 ipykernel==6.29.5 ipython==9.0.2 ipython_pygments_lexers==1.1.1 jedi==0.19.2 Jinja2==3.1.6 joblib==1.4.2 jupyter_client==8.6.3 jupyter_core==5.7.2 kiwisolver==1.4.8 llvmlite==0.44.0 lppls==0.6.20 lxml==5.4.0 markdown-it-py==3.0.0 MarkupSafe==3.0.2 matplotlib==3.10.1 matplotlib-inline==0.1.7 mdurl==0.1.2 mpmath==1.3.0 multiprocess==0.70.18 multitasking==0.0.11 narwhals==1.33.0 nest_asyncio==1.6.0 networkx==3.4.2 numba==0.61.2 numpy==2.2.4 openpyxl==3.1.5 osqp==1.0.4 outcome==1.3.0.post0 packaging==24.2 pandas==2.2.3 pandas-datareader==0.10.0 parso==0.8.4 pathos==0.3.4 patsy==1.0.1 peewee==3.17.9 pexpect==4.9.0 pickleshare==0.7.5 pillow==11.1.0 pip==25.0 platformdirs==4.3.7 pox==0.3.6 ppft==1.7.7 prompt_toolkit==3.0.50 protobuf==6.31.1 psutil==5.9.0 ptyprocess==0.7.0 pure_eval==0.2.3 pyarrow==20.0.0 pycparser==2.22 Pygments==2.19.1 pyparsing==3.2.1 PySocks==1.7.1 python-dateutil==2.9.0.post0 pytz==2025.1 PyYAML==6.0.2 pyzmq==26.2.0 requests==2.32.3 rich==14.0.0 scikit-learn==1.6.1 scipy==1.15.2 scs==3.2.7.post2 seaborn==0.13.2 selenium==4.33.0 setuptools==75.8.0 six==1.17.0 sniffio==1.3.1 sortedcontainers==2.4.0 soupsieve==2.6 stack_data==0.6.3 statsmodels==0.14.4 sympy==1.13.1 threadpoolctl==3.6.0 torch==2.6.0 torchvision==0.21.0 tornado==6.4.2 tqdm==4.67.1 traitlets==5.14.3 trio==0.30.0 trio-websocket==0.12.2 typing_extensions==4.13.2 tzdata==2025.1 urllib3==2.4.0 wcwidth==0.2.13 websocket-client==1.8.0 websockets==15.0.1 wheel==0.45.1 wsproto==1.2.0 xarray==2025.6.1 xyzservices==2025.1.0 yfinance==0.2.63 zipp==3.21.0
        This will open the Jupyter Notebook interface in your web browser, allowing you to navigate to the chapter directories and run the provided notebooks.

    Alternatively, you can use JupyterLab:

    Each chapter's solutions are contained within its respective directory. Simply navigate to the desired chapter and open the Jupyter notebook to explore the solutions. Execute the cells within the notebook to run the code and see the results.

bash
cd chapter2
jupyter notebook chapter2_solution.ipynb # or jupyter lab chapter2_solution.ipynb
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Write your code and tests.
4.  Submit a pull request.

> Contributions should follow PEP 8 guidelines and include unit tests where applicable. Ensure your code is well-documented and easy to understand.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.