# DTNN Streamlit Online

This folder is the Python 3.11 deployment package for the RF and Q-RF
prediction website. It uses the original TensorFlow neural-network architecture
and the original 1% and 2% TensorFlow checkpoints. The training project in
`Hyperelastic` is not required by the deployed website and is not modified.

## Included runtime

- Python 3.11
- Streamlit 1.59.2
- TensorFlow CPU 2.15.1 through `tensorflow.compat.v1`
- NumPy 1.26.4
- Original DTNN architecture: `4 -> 64 -> 64 -> 128 -> 128 -> 1`

`dtnn_web_model.py` contains only the TensorFlow graph and checkpoint-loading
code needed by the website. It does not retrain or approximate the models.

## Run locally with Python 3.11

From this directory in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tests\test_model_outputs.py
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The regression test compares both migrated checkpoint loaders with predictions
previously saved by the original TensorFlow 1.x project.

## Deploy to Streamlit Community Cloud

1. Put the contents of this folder at the root of a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Select `streamlit_app.py` as the entrypoint.
4. Open Advanced settings and select Python 3.11.
5. Deploy and run several predictions with both model selections.

All model and asset paths are resolved relative to `streamlit_app.py`, so the
app does not depend on the Cloud process's working directory.

## Model-selection behavior

- **2% model:** use the 2% checkpoint prediction directly.
- **1% model:** evaluate both checkpoints and use the 1% result unless it is
  greater than the 2% result, in which case the 2% result is used as the
  conservative cap.

