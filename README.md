# Suicide Ideation Detection — A Comparative Study of Open-Source LLMs

Research prototype comparing six open-source decoder-based LLMs, fine-tuned with LoRA,
for binary suicide ideation classification on the Kaggle SuicideWatch dataset.

**This is a research prototype, not a diagnostic or clinical tool.**

## Repository Structure
notebooks/ Kaggle training notebooks, one per candidate model
app/ Streamlit research-prototype interface

## Models Compared

| Model | Parameters | Fine-tuning |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.5B | LoRA |
| Phi-2 | 2.7B | LoRA |
| OLMo-2-0425-1B-Instruct | 1B | LoRA |
| Gemma-3-1B-it | 1B | LoRA |
| Llama-3.2-1B-Instruct | 1B | LoRA |
| SmolLM2-1.7B-Instruct | 1.7B | LoRA |

All six models were selected for being absent or under-represented in the
literature on suicide-risk detection, and were fine-tuned under an identical
protocol (same fixed data split, same hyperparameters) for a controlled
comparison.

## Results

| Rank | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | Phi-2 | 0.9847 | 0.9815 | 0.9880 | 0.9847 |
| 2 | OLMo-2-1B-Instruct | 0.9840 | 0.9821 | 0.9860 | 0.9840 |
| 3 | Qwen2.5-0.5B-Instruct | 0.9840 | 0.9840 | 0.9840 | 0.9840 |
| 4 | Gemma-3-1B-it | 0.9837 | 0.9846 | 0.9827 | 0.9837 |
| 5 | Llama-3.2-1B-Instruct | 0.9833 | 0.9885 | 0.9780 | 0.9832 |
| 6 | SmolLM2-1.7B-Instruct | 0.9830 | 0.9820 | 0.9840 | 0.9830 |

## Reproducing the Training

Each notebook in `notebooks/` is self-contained and was run independently on
Kaggle (single T4 GPU). To reproduce:

1. Open a notebook on Kaggle, attach the `nikhileswarkomati/suicide-watch` dataset.
2. Run all cells with `RUN_MODE = "smoke"` first to verify no errors.
3. Set `RUN_MODE = "full"` and run via "Save & Run All (Commit)".
4. The resulting `final_model_<name>/` folder and `results/<name>.json`
   metrics file are produced in the notebook's output.

## Running the Streamlit Application

1. Download a `final_model_<name>/` folder from a completed training run.
2. Place it in the `app/` folder, next to `app.py`.
3. Windows: double-click `run_app.bat` (creates a virtual environment and
   launches the app automatically on first run).
   Manual alternative:
pip install -r requirements.txt
streamlit run app.py


