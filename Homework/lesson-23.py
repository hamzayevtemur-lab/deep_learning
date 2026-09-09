import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

### Task 1: Using the pipeline API
print("\n1. Sentiment Analyssi")

sentiment_pipeline=pipeline("sentiment-analysis") # type:ignore

sentiment_texts=[
    "I really enjoyed this movie.",
    "This product is terrible and disappointing.",
    "The service was excellent."
]

sentiment_results=sentiment_pipeline(sentiment_texts)
for text, result in zip(sentiment_texts, sentiment_results):
    print(f"Text: {text}")
    print(f"Prediction:{result}")
 
### Text Generation
print("\n2. Text Generation")

text_generator = pipeline(
    "text-generation",
    model="distilgpt2"
) 

generation_result=text_generator(
    "Deep Learning is",
    max_new_tokens=30,
    num_return_sequences=1
) 

print(generation_result[0]["generated_text"])


### Zero-Shot Classification
print("\n3. Zero-Shot Classification")

zero_shot_classifier = pipeline(
    "zero-shot-classification"
)

text="I bought a new laptop and I am very happy with its performance."

candidate_labels=[
    "technology",
    "sports",
    "food",
    "politics"
]

zero_shot_result=zero_shot_classifier(
    text, candidate_labels
)

print("Text:", text)
print("Labels:", zero_shot_result["labels"]) #type: ignore
print("Scores:", zero_shot_result["scores"]) #type: ignore





### Task 2: Manually load tokenizer and model
print("\n4. Manual Model and Tokenizer")

model_name="distilbert-base-uncased-finetuned-sst-2-english"

tokenizer=AutoTokenizer.from_pretrained(
    model_name
)

model=AutoModelForSequenceClassification.from_pretrained(
    model_name
)

print("Model:", model_name)
print("Number of labels:", model.config.num_labels)
print("ID to Label:", model.config.id2label)


### Task 3: Preprocess raw text
print("\n5. Tokenization")

texts = [
    "I really like this movie.",
    "The movie was boring and disappointing.",
    "The acting was excellent."
]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print("Input IDs:")
print(inputs["input_ids"])

print("\nAttention Mask:")
print(inputs["attention_mask"])

print("\nInput Tensor Shape:")
print(inputs["input_ids"].shape)


### Task 4: Forward pass
print("\n6. Forward Pass")

model.eval()

with torch.no_grad():
    outputs=model(**inputs)
    
logits=outputs.logits()

print("Raw Logits:")
print(logits)

print("\nLogits Shape:")
print(logits.shape)

### Task 5: Post-process the output
print("\n7. Predictions")

probabilities=torch.softmax(
    logits, dim=-1
)

predicted_class_ids=torch.argmax(
    probabilities, dim=-1
)

for i, text in enumerate(texts):
    predicted_id = predicted_class_ids[i].item()
    predicted_label = model.config.id2label[predicted_id]
    confidence = probabilities[i, predicted_id].item()

    print(f"\nText: {text}")
    print(f"Predicted Class ID: {predicted_id}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")


