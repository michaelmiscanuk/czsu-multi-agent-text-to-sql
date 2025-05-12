@echo off

:: Install requirements
pip install -r requirements.txt

:: Install jupyter notebook and extension packages compatible with modern Jupyter
pip install notebook>=7.0.0 jupyterlab ipywidgets

:: Create jupyter config directory if it doesn't exist
mkdir %USERPROFILE%\.jupyter 2>nul

:: Enable ipywidgets extension using the correct method for modern Jupyter
pip install jupyterlab-widgets
jupyter labextension list

:: For classic notebook support - fixing the syntax error in the Python command
pip install jupyter_nbextensions_configurator 

echo Setup complete! Environment is ready for use.