import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer

# 加载训练数据、分词器、预训练模型以及评价方法
dataset = load_dataset('glue', 'mrpc')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-mini', return_dict=True)
metric = load_metric('glue', 'mrpc')

# 对训练集进行分词
def tokenize(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding="max_length", max_length = 100)
toked_dataset = dataset.map(tokenize, batched=True)
encoded_dataset = toked_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

# 将数据集格式化为torch.Tensor类型以训练PyTorch模型
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
encoded_dataset.set_format(type='torch', columns=columns)

# 定义评价指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)

# 定义训练参数TrainingArguments，默认使用AdamW优化器
args = TrainingArguments(
    "no-need-to-care-about-this-folder",
    evaluation_strategy="epoch",        # 定义每轮结束后进行评价
    learning_rate=3e-5,                 # 定义初始学习率
    per_device_train_batch_size=32,     # 定义训练批次大小
    per_device_eval_batch_size=32,      # 定义测试批次大小
    num_train_epochs=2,                 # 定义训练轮数
)

# 定义Trainer，指定模型和训练参数，输入训练集、验证集、分词器以及评价函数
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

print("\nSaving model to bert-for-mrpc")
model.save_pretrained("./bert-for-mrpc")
model = BertForSequenceClassification.from_pretrained("./bert-for-mrpc", return_dict=True)

preds = []
print("\nPractical task examples:\n")
example = dataset["train"][25:30]
example_loader = DataLoader(encoded_dataset["train"], 5)
with torch.no_grad():
    for i, batch in enumerate(example_loader):
        if i == 5:
            pred = model(**batch)["logits"]
            preds.extend(pred.squeeze().argmax(-1).tolist())
            break

texts = []
for key in example:
    if type(example[key][0]) == str:
        texts.append(example[key])
for i in range(len(preds)):
    for text in texts:
        print(text[i], end = '     ')
    print(f"\nPrediction: {preds[i]}, {'which is correct.' if preds[i] == example['label'][i] else 'which is wrong.'}\n")