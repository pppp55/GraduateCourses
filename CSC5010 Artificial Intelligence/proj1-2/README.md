# CSC5010 Project 1-2 Implementation Notes

This folder now contains runnable evaluation scripts for the CBOW and Skip-Gram
Word2Vec embeddings required by the assignment.

## Project structure

- `data/`: assignment evaluation datasets.
- `docs/`: original assignment PDFs, DOCX, and readme.
- `embeddings/`: put trained `cbow.vec` and `skipgram.vec` here.
- `scripts/`: training-code downloader and all evaluation scripts.
- `report/`: report template and report working files.
- `results/`: generated CSV files, analogy plot data, and plot PNGs.
- `training_code/`: downloaded HIT-SCIR `chp5` training code.

## 1. Train the two models

Download the training code:

```bash
conda run -n normal python scripts/prepare_training_code.py
```

Then run the assignment training scripts from `training_code/`:

```bash
cd training_code
conda run -n normal python cbow.py
conda run -n normal python skipgram.py
```

Copy the generated `cbow.vec` and `skipgram.vec` files into `embeddings/`.

## 2. Run all evaluations

Use your student ID as the seed so the randomly selected report examples are
reproducible:

```bash
conda run -n normal python scripts/run_all_evaluations.py --student-id YOUR_STUDENT_ID
```

Outputs are written to `results/`.

## 3. Output files

- `*_knn_all.csv`: top-10 nearest words for all 50 query words.
- `*_knn_summary.csv`: average top-10 similarity for each query word.
- `*_knn_detail_words.csv`: the four seed-selected words to discuss in detail.
- `*_simlex_all.csv`: all SimLex-999 covered and uncovered word pairs.
- `*_simlex_sample_20.csv`: the 20 seed-selected SimLex examples for the report.
- `*_analogy_all.csv`: every analogy question, prediction, and correctness flag.
- `*_analogy_by_category.csv`: per-category analogy accuracy.
- `*_analogy_sample_10.csv`: the 10 seed-selected analogy examples for the report.
- `*_analogy_plots/`: vector plot PNGs when `matplotlib` is installed, otherwise CSV
  files containing the same vector values.

## 4. Report checklist

- Explain CBOW and Skip-Gram training settings and include the generated `.vec`
  files in the final zip.
- Compare KNN average similarities and discuss the four detailed query words.
- Report SimLex Spearman correlation for both models and discuss the 20 sampled
  word pairs.
- Report analogy accuracy overall and by category, include 10 sampled examples,
  and include the vector plots or plot data for the selected categories.
