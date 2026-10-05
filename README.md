# Suicide Ideation Detection - Open-Source LLM Comparison

This repository contains the code and notebooks used to compare six open-source
decoder-based LLMs for binary suicide ideation classification on the Kaggle
SuicideWatch dataset.

The models were fine-tuned and evaluated using the same data split
and training setup.

## Repository Structure

- `notebooks/` — Kaggle notebooks used to train and evaluate the six models.
- `app/` — Streamlit application for testing the selected model.

## Models

| Model | Parameters | Fine-tuning |
|---|---:|---|
| Qwen2.5-0.5B-Instruct | 0.5B | LoRA |
| Phi-2 | 2.7B | LoRA |
| OLMo-2-0425-1B-Instruct | 1B | LoRA |
| Gemma-3-1B-it | 1B | LoRA |
| Llama-3.2-1B-Instruct | 1B | LoRA |
| SmolLM2-1.7B-Instruct | 1.7B | LoRA |

The six models were chosen because they are relatively under-represented in
previous work on suicide-risk detection using this dataset. All models were
trained using the same main settings to make the comparison as consistent as
possible.

## Results

| Rank | Model | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| 1 | Phi-2 | 0.9847 | 0.9815 | 0.9880 | 0.9847 |
| 2 | OLMo-2-1B-Instruct | 0.9840 | 0.9821 | 0.9860 | 0.9840 |
| 3 | Qwen2.5-0.5B-Instruct | 0.9840 | 0.9840 | 0.9840 | 0.9840 |
| 4 | Gemma-3-1B-it | 0.9837 | 0.9846 | 0.9827 | 0.9837 |
| 5 | Llama-3.2-1B-Instruct | 0.9833 | 0.9885 | 0.9780 | 0.9832 |
| 6 | SmolLM2-1.7B-Instruct | 0.9830 | 0.9820 | 0.9840 | 0.9830 |

## Training

The training notebooks are in the `notebooks/` folder. Each notebook
corresponds to one of the six models and was run on Kaggle using a single
T4 GPU.

To run a notebook:

1. Open the notebook on Kaggle.
2. Add the `nikhileswarkomati/suicide-watch` dataset.
3. Run the notebook with `RUN_MODE = "smoke"` first to check that the setup works.
4. Set `RUN_MODE = "full"`.
5. Run the notebook using **Save & Run All (Commit)**.

After training, the notebook produces a `final_model_<name>/` folder and a
`results/<name>.json` file containing the evaluation results.

## Streamlit Application

The `app/` folder contains the Streamlit interface using the selected model.

To run the application:

1. Copy a `final_model_<name>/` folder from a completed training run into
   the `app/` folder.
2. Make sure it is located next to `app.py`.
3. On Windows, double-click `run_app.bat`. The script creates the virtual
   environment and starts the application on the first run.

You can also start the application manually:

```bash
pip install -r requirements.txt
streamlit run app.py
