@echo off
echo Installing Security TTP RAG Model Dependencies
echo ============================================

echo.
echo Creating virtual environment...
python -m venv venv

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To activate the environment, run: venv\Scripts\activate
echo To run the training pipeline: python train_pipeline.py
echo To run inference: python src\inference.py
echo To open the Jupyter notebook: jupyter notebook Security_TTP_RAG_Training.ipynb

pause
