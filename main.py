from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
import os
import unicodedata
from fastapi.middleware.cors import CORSMiddleware

###############################################################################
# General Explanation:
# This code sets up a FastAPI application that provides two main endpoints:
# 1. /train-classifier: Train a text classifier model on labeled articles.
# 2. /classify-articles: Use the trained model to classify new articles based
#    on their content, returning only those deemed "relevant" above a certain 
#    probability threshold.
#
# Key steps:
# - Preprocessing text and converting it into numerical form (tokenization).
# - Using a pre-trained language model (DistilBERT) from Hugging Face Transformers.
# - Fine-tuning that model on a custom dataset of articles labeled as relevant or not.
# - Saving the model locally so it can be reused for classification requests.
###############################################################################


###############################################################################
# FastAPI and CORS Setup
###############################################################################

app = FastAPI()

# origins: Allowed front-end URLs that can interact with this backend via API calls.
# This is needed to avoid CORS (Cross-Origin Resource Sharing) issues, ensuring that
# your frontend (for example, a React or Next.js application on localhost:3000) can
# communicate with this backend.
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

# Add CORS middleware to the FastAPI app so that browsers won't block requests
# from these origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


###############################################################################
# Text Cleaning Function
###############################################################################
def clean_text(text):
    """
    Normalizes text by:
    1. Checking if text is None or not a string, converting to string if needed.
    2. Using Unicode normalization (NFKD) to break down characters into basic forms.
    3. Encoding into ASCII and ignoring characters that can't be converted, 
       stripping special characters and accents.
    4. Returning a cleaned ASCII-only version of the text.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    normalized_text = unicodedata.normalize("NFKD", text)
    ascii_text = normalized_text.encode("ascii", errors="ignore").decode("utf-8")
    return ascii_text.strip()


###############################################################################
# Model and Tokenizer Setup
###############################################################################

# Paths to store the trained model and tokenizer.
MODEL_PATH = "custom_classifier"
TOKENIZER_PATH = "custom_tokenizer"

# We use a pre-trained model from Hugging Face Transformers ("distilbert-base-uncased").
# DistilBERT is a lightweight version of BERT, a popular Transformer-based NLP model.
# "uncased" means it doesn't distinguish between uppercase and lowercase letters.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# If a model is already trained and saved, load it. Otherwise, initialize a new model.
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    # Loading previously fine-tuned model and tokenizer from disk.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print("Model and tokenizer loaded from local storage.")
else:
    # Loading a pre-trained DistilBERT model for sequence classification with 2 labels.
    # num_labels=2 usually means we have a binary classification task: relevant vs not relevant.
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    print("Initialized new model and tokenizer.")


###############################################################################
# Pydantic Models for Request Validation
###############################################################################
class TrainingArticle(BaseModel):
    title: Optional[str] = ""
    description: Optional[str] = ""
    content: Optional[str] = ""
    label: int  # 0 or 1
    
class TrainingDataWithRewrite(BaseModel):
    """
    Request body for the training endpoint:
    - articles: List of labeled articles
    - rewrite: If True, overwrite the existing JSON dataset instead of merging
    """
    articles: List[TrainingArticle]
    rewrite: bool = False  # default is False

class ClassificationData(BaseModel):
    # For classification, we receive articles without the need for labels,
    # since we are predicting them. The threshold is a probability cutoff
    # used to decide if an article is "relevant".
    articles: List[Dict]
    threshold: float = 0.5


###############################################################################
# Custom Dataset for the Trainer
###############################################################################
class CustomDataset(Dataset):
    """
    A PyTorch Dataset object that returns training examples for the Trainer.
    Each item includes:
    - input_ids: Numerical tokens that represent the text.
    - attention_mask: A mask telling the model which tokens are actual text
      and which are padding (so it can ignore padding).
    - labels: The ground-truth label for the article (e.g., 0 or 1).
    """

    def __init__(self, encodings, labels):
        # encodings: a dictionary with 'input_ids' and 'attention_mask' as tensors.
        # labels: a tensor of labels corresponding to each input example.
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        # The length of the dataset is the number of examples.
        return len(self.labels)

    def __getitem__(self, idx):
        # Given an index, return a single training example as a dictionary.
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


###############################################################################
# Training Endpoint
###############################################################################
@app.post("/train-classifier")
def train_classifier(data: TrainingDataWithRewrite):
    """
    This endpoint is called to train the classifier using provided labeled articles.
    Steps:
    1. Extract text fields (title, description, content) from each article and combine them.
    2. Clean and normalize the text.
    3. Assign 'labels' from each article (0 or 1).
    4. Tokenize texts: Convert from raw text to token IDs that the model understands.
    5. Split the data into training and validation sets (80/20).
    6. Create PyTorch Dataset objects for training and validation.
    7. Define training arguments (like epochs, batch size, learning rate).
    8. Use Hugging Face 'Trainer' to fine-tune the model.
    9. Save the trained model and tokenizer locally for future use.
    """

    # Extract and preprocess text
    texts = []
    labels = []
    for article in data.articles:
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")
        # Combine relevant fields into one string. The model will learn from the combined info.
        text = f"Title: {clean_text(title)}. Description: {clean_text(description)}. Content: {clean_text(content)}."
        texts.append(text)
        labels.append(article["label"])

    # Convert labels to a torch tensor
    labels = torch.tensor(labels)

    # Tokenize all texts. The tokenizer converts words to IDs and 
    # handles special tokens and padding/truncation.
    # We do this without return_tensors="pt" because we want 
    # to manually manage train/test splits first.
    all_encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

    # Split the dataset into training and validation sets using sklearn's train_test_split.
    # This helps to monitor the model's performance on unseen data during training.
    indices = list(range(len(labels)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    def select_subset(encodings, subset_indices):
        # This function selects a subset of the encodings based on the given indices.
        # We then convert them to tensors.
        return {
            "input_ids": torch.tensor([encodings["input_ids"][i] for i in subset_indices]),
            "attention_mask": torch.tensor([encodings["attention_mask"][i] for i in subset_indices])
        }

    # Create subsets for train and validation
    train_enc = select_subset(all_encodings, train_indices)
    val_enc = select_subset(all_encodings, val_indices)
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    # Create PyTorch Dataset objects for Trainer
    train_dataset = CustomDataset(train_enc, train_labels)
    val_dataset = CustomDataset(val_enc, val_labels)

    # Define training arguments.
    # Important parameters:
    # - num_train_epochs: number of times we pass through the entire dataset.
    # - per_device_train_batch_size: how many samples per training step.
    # - evaluation_strategy='epoch': evaluate the model each epoch.
    # - save_strategy='epoch': save model each epoch.
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
    )

    # Data collator: automatically pads the input sequences to the same length.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # The Trainer class simplifies the training loop:
    # It handles feeding data to the model, backpropagation, evaluation, and more.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Save the trained model and tokenizer for future classification
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_PATH)

    return {"message": "Classifier trained and saved successfully."}


###############################################################################
# Classification Endpoint
###############################################################################
@app.post("/classify-articles")
def classify_articles(data: ClassificationData):
    """
    This endpoint is called to classify new articles using the trained model.
    Steps:
    1. Load the previously trained model and tokenizer from disk.
    2. Process the input articles similarly to training: combine title, description,
       and content into one text string and clean it.
    3. Tokenize the input texts.
    4. Run the model in evaluation mode to get predictions.
    5. Apply a softmax to convert logits (raw model outputs) to probabilities.
    6. Filter articles that have a probability for the "relevant" class (index 1) above the threshold.
    7. Return these filtered articles along with their probabilities.
    """

    # Ensure we have a trained model to classify with.
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        raise HTTPException(status_code=400, detail="No trained model found. Train the classifier first.")

    # Load the saved model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Preprocess articles similarly to training step
    texts = []
    for article in data.articles:
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")
        text = f"Title: {clean_text(title)}. Description: {clean_text(description)}. Content: {clean_text(content)}."
        texts.append(text)

    # Tokenize new articles
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

    # Put model in evaluation mode (disables dropout layers for consistent results)
    model.eval()
    with torch.no_grad():
        # Get the raw output logits from the model for each article.
        outputs = model(**encodings)
        # outputs.logits are the raw, unnormalized scores for each class (0 or 1).
        # Apply softmax to get probabilities between 0 and 1 that sum to 1.
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Filter out only the articles where the probability of class '1' (relevant) is greater than threshold.
    # class 0 = not relevant, class 1 = relevant, so we check prob[1] > threshold.
    classified_articles = [
        {**article, "probability": float(prob[1])}
        for article, prob in zip(data.articles, probabilities)
        if prob[1] > data.threshold
    ]

    # Return the relevant articles with their probabilities.
    return {"classified_articles": classified_articles}
