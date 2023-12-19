from datasets import load_dataset
import random
from PIL import ImageDraw, ImageFont, Image
from transformers import ViTImageProcessor
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
from datasets import load_metric
from transformers import ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

random.seed(123)

# ds = load_dataset('HugsVision/STANFORD-CARS-TYPES').shuffle(123)
ds = load_dataset('HugsVision/SkinDisease').shuffle(123)

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

def transform(example_batch):
  inputs = processor([x.convert("RGB").resize((400,400)) for x in example_batch['image']], return_tensors='pt')

  inputs['labels'] = example_batch['labels']
  return inputs

prepared_ds = ds.with_transform(transform)

def collate_fn(batch):
  return {
      'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
      'labels': torch.tensor([x['labels'] for x in batch])
  }

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

labels_names = ds['train'].features['labels'].names

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels_names),
    id2label={str(i): c for i, c in enumerate(labels_names)},
    label2id={c: str(i) for i, c in enumerate(labels_names)}
)

training_args = TrainingArguments(
    output_dir="./skin-disease-classification",
    per_device_train_batch_size=24,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=150,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

splitted_ds = prepared_ds["train"].train_test_split(test_size=0.2)
train_ds = splitted_ds["train"]
print(train_ds)
validation_ds = splitted_ds["test"]
print(validation_ds)
test_ds = prepared_ds["test"]
print(test_ds)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=validation_ds,
    tokenizer=processor,
)

train_results = trainer.train()
trainer.save_model()

print("***** Starting Evaluation *****")
metrics = trainer.evaluate(test_ds)
print(metrics)

predictions, labels, _ = trainer.predict(test_ds)
predictions = np.argmax(predictions, axis=1)

f1_score = classification_report(
    labels,
    predictions,
    digits=4,
    target_names=labels_names,
)
print(f1_score)

cm = confusion_matrix(labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("./confusion_matrix_ViT.png")

