# CSC5010 Project 3 Report: Transformer Text Classification

Student ID: 225040065

## 1. Project Goal

This project runs a Transformer encoder classifier on a truncated Chinese THUCNews dataset. The task is to classify Chinese news titles into 10 categories: 财经, 房产, 股票, 教育, 科技, 社会, 时政, 体育, 游戏, and 娱乐.

The grading requirements define two experiments. Task A uses the original text without preprocessing. Task B applies text preprocessing before training and testing. Both tasks must output running results and misclassified documents, then compare cases where one task is correct and the other is wrong.

## 2. Dataset and Model

- Training set: 45,000 titles
- Development set: 2,500 titles
- Test set: 2,500 titles
- Vocabulary limit: 10,000 characters plus `<UNK>` and `<PAD>`
- Sequence length: 32 characters
- Model: Transformer encoder classifier
- Epochs: 6
- Seed: derived from student ID 225040065

The model uses a character-level tokenizer. Each title is padded or truncated to 32 characters. The Transformer includes token embedding, sinusoidal positional encoding, multi-head self-attention, feed-forward layers, residual connections, layer normalization, and a final linear classifier.

## 3. Task A: Original Text

Task A uses the original dataset directly. No punctuation or whitespace is removed before building the vocabulary or encoding the train/dev/test sets.

- Test loss: 0.493174
- Test accuracy: 0.846800
- Misclassified test documents: 383

```text
Test Accuracy: 0.846800
Test Loss: 0.493174
```

The full Task A classification report and confusion matrix are saved in `outputs/taskA/taskA_metrics.txt`. All predictions are saved in `outputs/taskA/taskA_predictions.csv`, and all wrong predictions are saved in `outputs/taskA/taskA.misclassified.csv`.

## 4. Task B: Preprocessed Text

Task B removes Chinese and English punctuation and normalizes whitespace before vocabulary construction and encoding. The labels and train/dev/test split are unchanged. This keeps the experiment focused on the effect of text cleaning.

- Test loss: 0.570312
- Test accuracy: 0.842400
- Misclassified test documents: 394

```text
Test Accuracy: 0.842400
Test Loss: 0.570312
```

The full Task B classification report and confusion matrix are saved in `outputs/taskB/taskB_metrics.txt`. All predictions are saved in `outputs/taskB/taskB_predictions.csv`, and all wrong predictions are saved in `outputs/taskB/taskB.misclassified.csv`.

## 5. Task A vs. Task B Comparison

| Task | Preprocessing | Test accuracy | Misclassified documents |
|---|---|---:|---:|
| Task A | None | 0.846800 | 383 |
| Task B | Remove punctuation / normalize whitespace | 0.842400 | 394 |

Task A wrong but Task B correct: 138 documents.
Task B wrong but Task A correct: 149 documents.

## 6. Three Documents Wrong in Task A but Correct in Task B

| Test index | Document | True label | Task A prediction | Task B prediction |
|---:|---|---|---|---|
| 5 | 本科未录取还有这些路可以走 | 教育 | 房产 | 教育 |
| 28 | 调查显示：29.5%的人不满意当年所选高考专业 | 教育 | 股票 | 教育 |
| 105 | 五W让你提高英语四六级听力 | 教育 | 股票 | 教育 |

Brief explanations:

1. `本科未录取还有这些路可以走`: The cleaned version keeps the main words but changes token positions after normalization, which can alter attention patterns.
2. `调查显示：29.5%的人不满意当年所选高考专业`: Task B removes 3 punctuation/space characters, making the core keywords more concentrated for the character-level Transformer.
3. `五W让你提高英语四六级听力`: The cleaned version keeps the main words but changes token positions after normalization, which can alter attention patterns.

## 7. Three Documents Wrong in Task B but Correct in Task A

| Test index | Document | True label | Task A prediction | Task B prediction |
|---:|---|---|---|---|
| 43 | 教育话题：不能让何川洋做“牺牲品” | 教育 | 教育 | 财经 |
| 67 | 名校农村生比例降低 该酝酿“平权法案”了 | 教育 | 教育 | 社会 |
| 89 | 浙商大法语系介绍：半数毕业生出国深造 | 教育 | 教育 | 社会 |

Brief explanations:

1. `教育话题：不能让何川洋做“牺牲品”`: The removed punctuation may have carried useful title structure; after cleaning, 3 characters are removed and the model loses some boundary cues.
2. `名校农村生比例降低 该酝酿“平权法案”了`: The removed punctuation may have carried useful title structure; after cleaning, 3 characters are removed and the model loses some boundary cues.
3. `浙商大法语系介绍：半数毕业生出国深造`: The removed punctuation may have carried useful title structure; after cleaning, 1 characters are removed and the model loses some boundary cues.

## 8. Output Files

- `outputs/taskA/taskA_metrics.txt`: Task A running logs, loss, accuracy, classification report, and confusion matrix.
- `outputs/taskB/taskB_metrics.txt`: Task B running logs, loss, accuracy, classification report, and confusion matrix.
- `outputs/taskA/taskA_predictions.csv`: all Task A test predictions.
- `outputs/taskB/taskB_predictions.csv`: all Task B test predictions.
- `outputs/taskA/taskA.misclassified.csv`: all Task A misclassified test documents.
- `outputs/taskB/taskB.misclassified.csv`: all Task B misclassified test documents.
- `outputs/taskA_wrong_taskB_correct.csv`: documents fixed by preprocessing.
- `outputs/taskB_wrong_taskA_correct.csv`: documents hurt by preprocessing.

## 9. Conclusion

The project successfully runs the Transformer classifier for both required tasks and outputs the requested misclassified documents. Task B tests whether punctuation removal and whitespace normalization improve the model. Comparing the cross-corrected documents shows that preprocessing can help when punctuation distracts from core category words, but it can also hurt when punctuation provides useful boundary or title-structure cues. The final deliverable includes this report, all source code, all output results, and the generated comparison files.