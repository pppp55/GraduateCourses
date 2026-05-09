# CSC5010 Project 4 Report: BERT Fine-tuning

Student ID: 225040065

## 1. Project Goal

This project fine-tunes a small BERT model on two GLUE benchmark tasks. Task 1 is SST-2 single-sentence sentiment analysis, where each movie review sentence is classified as negative or positive. Task 2 is MRPC sentence-pair paraphrase analysis, where each pair is classified as not paraphrase or paraphrase.

The implementation uses the provided project templates as the task definition, but updates the runner for the current Python environment. In particular, the `datasets.load_metric` API used by the original templates is no longer available in the installed `datasets` version, so the final metrics are computed with `sklearn.metrics`.

## 2. Environment and Model

- Model: `prajjwal1/bert-mini`
- Tokenizer: `bert-base-uncased`
- Epochs: 2.0
- Batch size: 32
- Random seed: 225040065
- Framework: HuggingFace Transformers Trainer with PyTorch

Both tasks use `bert-base-uncased` tokenization and the lightweight `prajjwal1/bert-mini` sequence classification model. SST-2 is encoded as a single sentence with maximum length 64. MRPC is encoded as a sentence pair with maximum length 100.

## 3. Final Evaluation Results

| Task | Validation examples | Accuracy | F1 | Correct | Incorrect |
|---|---|---|---|---|---|
| SST-2 sentiment | 872 | 0.8521 | N/A | 743 | 129 |
| MRPC paraphrase | 408 | 0.7475 | 0.8352 | 305 | 103 |

The final evaluation numbers are taken from the validation split because the project handout states that the test set is not available for this assignment.

## 4. SST-2 Practical Examples

| Index | Sentence | Gold | Prediction | Confidence | Correct |
|---|---|---|---|---|---|
| 253 | if you believe any of this , i can make you a real deal on leftover enron stock that will double in value a week from friday .  | negative | positive | 0.6608 | 0 |
| 468 | the characters are interesting and often very creatively constructed from figure to backstory .  | positive | positive | 0.9901 | 1 |
| 482 | cq 's reflection of artists and the love of cinema-and-self suggests nothing less than a new voice that deserves to be considered as a possible successor to the best european directors .  | positive | negative | 0.7078 | 0 |
| 772 | not the kind of film that will appeal to a mainstream american audience , but there is a certain charm about the film that makes it a suitable entry into the fest circuit .  | positive | positive | 0.9527 | 1 |

Comments:

1. The prediction is incorrect: the wording is short or stylistically indirect, so the model assigns `positive` even though the gold label is `negative`. This is a common error when sentiment depends on context rather than an obvious positive or negative word.
2. The prediction is correct: the sentence contains sentiment cues that align with the gold label `positive`, so the fine-tuned BERT model maps it to `positive`.
3. The prediction is incorrect: the wording is short or stylistically indirect, so the model assigns `negative` even though the gold label is `positive`. This is a common error when sentiment depends on context rather than an obvious positive or negative word.
4. The prediction is correct: the sentence contains sentiment cues that align with the gold label `positive`, so the fine-tuned BERT model maps it to `positive`.

## 5. MRPC Practical Examples

| Index | Sentence 1 | Sentence 2 | Gold | Prediction | Confidence | Correct |
|---|---|---|---|---|---|---|
| 45 | Rumsfeld , who has been feuding for two years with Army leadership , passed over nine active-duty four-star generals . | Rumsfeld has been feuding for a long time with Army leadership , and he passed over nine active-duty four-star generals . | paraphrase | paraphrase | 0.7986 | 1 |
| 115 | A European Union spokesman said the Commission was consulting EU member states " with a view to taking appropriate action if necessary " on the matter . | Laos 's second most important export destination - said it was consulting EU member states ' ' with a view to taking appropriate action if necessary ' ' on the matter . | not paraphrase | paraphrase | 0.8271 | 0 |
| 194 | About two hours later , his body , wrapped in a blanket , was found dumped a few blocks away . | Then his body was dumped a few blocks away , found in a driveway on Argyle Road . | not paraphrase | paraphrase | 0.6422 | 0 |
| 325 | Some of the company 's software developers will join Microsoft , but details haven 't been finalized , said Mike Nash , corporate vice president of Microsoft 's security business unit . | Some of the companys software developers will join Microsoft , but details havent been finalized , said Mike Nash , corporate vice president of Microsofts security business unit . | paraphrase | paraphrase | 0.8324 | 1 |

Comments:

1. The prediction is correct: the pair has enough lexical/semantic evidence for `paraphrase`. Word overlap count is 16, and the model predicts `paraphrase`.
2. The prediction is incorrect: the pair may contain misleading word overlap or subtle differences in entities, numbers, or event details. The model predicts `paraphrase`, while the gold label is `not paraphrase`.
3. The prediction is incorrect: the pair may contain misleading word overlap or subtle differences in entities, numbers, or event details. The model predicts `paraphrase`, while the gold label is `not paraphrase`.
4. The prediction is correct: the pair has enough lexical/semantic evidence for `paraphrase`. Word overlap count is 23, and the model predicts `paraphrase`.

## 6. Output Files

- `outputs/sst2/sst2_metrics.txt`: SST-2 final metrics, classification report, and selected examples.
- `outputs/sst2/sst2_predictions.csv`: all SST-2 validation predictions.
- `outputs/sst2/sst2_examples.csv`: four SST-2 examples used in the report.
- `outputs/mrpc/mrpc_metrics.txt`: MRPC final metrics, classification report, and selected examples.
- `outputs/mrpc/mrpc_predictions.csv`: all MRPC validation predictions.
- `outputs/mrpc/mrpc_examples.csv`: four MRPC examples used in the report.
- `outputs/experiment_summary.json`: machine-readable summary of both experiments.

## 7. Conclusion

The BERT model was successfully fine-tuned and evaluated on both required NLP tasks. SST-2 tests single-sentence sentiment understanding, while MRPC tests sentence-pair semantic equivalence. The selected examples show that BERT can capture many direct sentiment and paraphrase cues, but it can still fail when wording is subtle, when surface overlap is misleading, or when a pair differs in small factual details.
