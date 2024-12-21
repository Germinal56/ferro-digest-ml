from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional, Union, Any
import unicodedata
import os
import json

# For embeddings
from sentence_transformers import SentenceTransformer

# For classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

###############################################################################
# FastAPI Setup
###############################################################################
app = FastAPI(
    title="Article Classification Service (Active Learning Demo)",
    description="""
A service to classify articles as "relevant" or "not relevant."
We store the labeled dataset in a JSON file (full_dataset.json)
to enable active learning (merging new data in).

Includes an endpoint to extract 'facts' from article content
(placeholder logic).
""",
    version="1.0.0",
)

# CORS (if needed)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
# Configuration / Paths
###############################################################################
DATASET_JSON_PATH = "data/full_dataset.json"
MODELS_DIR = "models"
EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "sbert_model")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "logistic_model.pkl"

)

###############################################################################
# Utility: Text Cleaning
###############################################################################
def clean_text(text: str) -> str:
    """
    Cleans and normalizes the text input:
    1. Check if text is None or not a string; convert if needed.
    2. Use Unicode normalization (NFKD) to remove diacritics.
    3. Encode to ASCII, ignoring chars that won't map, then decode.
    4. Strip leading/trailing spaces.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    normalized_text = unicodedata.normalize("NFKD", text)
    ascii_text = normalized_text.encode("ascii", errors="ignore").decode("utf-8", errors="ignore")
    return ascii_text.strip()

###############################################################################
# Pydantic Models for Classification
###############################################################################
class TrainingArticle(BaseModel):
    title: Optional[str] = ""
    description: Optional[str] = ""
    content: Optional[str] = ""
    source: Union[str, Dict[str, Any], None] = None
    label: int  # 0 or 1

class TrainingData(BaseModel):
    articles: List[TrainingArticle]

class ClassificationArticle(BaseModel):
    model_config = ConfigDict(extra='allow')  # Allows extra fields like 'publishedAt'
    title: Optional[str] = ""
    description: Optional[str] = ""
    content: Optional[str] = ""
    source: Union[str, Dict[str, Any], None] = None

class ClassificationData(BaseModel):
    articles: List[ClassificationArticle]
    threshold: float = 0.5

###############################################################################
# Global Model References
###############################################################################
embedding_model = None
classifier = None

###############################################################################
# Dataset Management
###############################################################################
def load_full_dataset() -> List[Dict]:
    """
    Loads the existing dataset (list of article dicts) from full_dataset.json.
    Returns an empty list if the file doesn't exist.
    """
    if os.path.exists(DATASET_JSON_PATH):
        with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return dataset  # Expecting a list[dict]
    else:
        return []

def save_full_dataset(articles_list: List[Dict]):
    """
    Saves the entire list of articles (list[dict]) to full_dataset.json,
    removing duplicates before writing.
    Overwrites any existing file with the new data.
    """
    os.makedirs(os.path.dirname(DATASET_JSON_PATH), exist_ok=True)

    # Remove duplicates by serializing each article to a JSON string.
    unique_map = {}
    for article in articles_list:
        article_key = json.dumps(article, sort_keys=True)
        unique_map[article_key] = article

    deduplicated_articles = list(unique_map.values())

    with open(DATASET_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(deduplicated_articles, f, indent=2)

###############################################################################
# Embedding Model / Classifier Loader
###############################################################################
def load_embedding_model():
    """
    Loads (or downloads) the SentenceTransformer, caching in a global variable.
    """
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
            print("Loaded local SentenceTransformer model.")
        except Exception:
            embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("Downloaded SentenceTransformer model. Saving locally...")
            os.makedirs(MODELS_DIR, exist_ok=True)
            embedding_model.save(EMBEDDING_MODEL_PATH)
    return embedding_model

def load_classifier(force_reload: bool = False):
    """
    Loads the trained LogisticRegression classifier from disk if it exists.
    If force_reload is True, reloads the classifier even if it's already loaded.
    """
    global classifier
    if classifier is None or force_reload:
        if os.path.exists(CLASSIFIER_PATH):
            classifier = joblib.load(CLASSIFIER_PATH)
            print("Classifier loaded from disk.")
        else:
            classifier = None
    return classifier

###############################################################################
# /train-classifier Endpoint (Full Retraining Each Time)
###############################################################################
class TrainingDataWithRewrite(BaseModel):
    """
    Additional field 'rewrite': if True, overwrite the existing dataset
    instead of merging.
    """
    articles: List[TrainingArticle]
    rewrite: bool = False

@app.post("/train-classifier")
def train_classifier(data: TrainingDataWithRewrite):
    """
    This endpoint merges new data (or overwrites existing data if 'rewrite=True') 
    in full_dataset.json, then does a full retraining of the LogisticRegression 
    model on the entire updated dataset.
    """
    global classifier

    # 1. Load embedding model
    model = load_embedding_model()
    print("Embedding model loaded.")

    # 2. Decide whether to overwrite or merge
    if data.rewrite:
        existing_dataset = []
        print("Rewrite=True: ignoring existing dataset.")
    else:
        existing_dataset = load_full_dataset()
        print(f"Existing dataset size: {len(existing_dataset)}")

    # 3. Convert new data to dicts
    new_articles = []
    for article in data.articles:
        new_articles.append({
            "title": article.title,
            "description": article.description,
            "content": article.content,
            "source": article.source,
            "label": article.label
        })
    print(f"New articles received: {len(new_articles)}")

    # 4. Merge or overwrite
    if data.rewrite:
        merged_dataset = new_articles
    else:
        merged_dataset = existing_dataset + new_articles
    print(f"Merged dataset size: {len(merged_dataset)}")

    # 5. Save merged dataset (deduplicates automatically)
    save_full_dataset(merged_dataset)
    print("Merged dataset saved to JSON.")

    # 6. Re-load updated dataset for training
    final_dataset = load_full_dataset()
    print(f"Final dataset size for training: {len(final_dataset)}")

    if not final_dataset:
        print("No data to train on.")
        return {"message": "No data available for training."}

    # 7. Prepare data
    texts, labels = [], []
    for item in final_dataset:
        title = item.get("title", "")
        description = item.get("description", "")
        content = item.get("content", "")
        source_field = item.get("source", "")
        if isinstance(source_field, dict):
            source = source_field.get("name", "")
        else:
            source = ""

        label = item.get("label", 0)

        combined_text = (
            f"Title: {clean_text(title)}. "
            f"Source: {clean_text(source)}. "
            f"Description: {clean_text(description)}. "
            f"Content: {clean_text(content)}."
        )
        texts.append(combined_text)
        labels.append(label)

    print(f"Prepared {len(texts)} items for training.")
    labels = np.array(labels)

    # 8. Embed
    print("Generating embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True)
    print("Embeddings generated.")

    # 9. Train/validation split
    if len(set(labels)) > 1:
        train_emb, val_emb, train_labels, val_labels = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"Training set size: {len(train_labels)}, Validation set size: {len(val_labels)}")
    else:
        train_emb, val_emb = embeddings, embeddings
        train_labels, val_labels = labels, labels
        print("Only one class or minimal data. Using entire dataset for training.")

    # 10. Train logistic regression
    print("Training the model...")
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        verbose=1
    )
    clf.fit(train_emb, train_labels)
    print("Model trained successfully.")

    # 11. Evaluate
    if len(set(val_labels)) > 1:
        val_preds = clf.predict(val_emb)
        report = classification_report(val_labels, val_preds, output_dict=True)
        print("Validation report generated.")
    else:
        report = {"warning": "Validation set has only one class. No detailed report."}
        print("Validation set had only one class.")

    # 12. Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, CLASSIFIER_PATH)
    classifier = clf  # Update global
    print("Model retrained and saved successfully.")

    return {
        "message": "Classifier trained successfully.",
        "classification_report": report
    }

###############################################################################
# /classify-articles Endpoint
###############################################################################
@app.post("/classify-articles")
def classify_articles(data: ClassificationData):
    """
    Classify new articles based on the trained logistic regression model.
    Returns only articles that meet the threshold (prob >= threshold).
    """
    # 1. Ensure we have a model
    load_embedding_model()
    clf = load_classifier(force_reload=True)
    if clf is None:
        raise HTTPException(status_code=400, detail="No trained classifier found. Train the classifier first.")

    # 2. Preprocess each article
    texts = []
    for article in data.articles:
        title = article.title or ""
        source_field = article.source or {}
        source = source_field.get("name", "") if isinstance(source_field, dict) else ""
        description = article.description or ""
        content = article.content or ""
        combined_text = (
            f"Title: {clean_text(title)}. "
            f"Source: {clean_text(source)}. "
            f"Description: {clean_text(description)}. "
            f"Content: {clean_text(content)}."
        )
        texts.append(combined_text)

    # 3. Generate embeddings
    model = load_embedding_model()
    embeddings = model.encode(texts, 
        convert_to_numpy=True, 
        show_progress_bar=True
    )

    # 4. Predict probabilities
    probs = clf.predict_proba(embeddings)[:, 1]
    threshold = data.threshold

    # 5. Build the response, returning only items with predicted_label=1
    classified_articles = []
    for article, prob in zip(data.articles, probs):
        predicted_label = 1 if prob >= threshold else 0
        # Overwrite 'label' and add 'probability'
        classified_article = {
            **article.model_dump(),
            "probability": float(prob),
            "label": predicted_label
        }
        if predicted_label == 1:
            classified_articles.append(classified_article)

    return {"classified_articles": classified_articles}

###############################################################################
# /extract-facts Endpoint (Placeholder)
###############################################################################
class FactExtractionArticle(BaseModel):
    title: Optional[str] = None
    full_content: str  # Full text of the article

class FactExtractionData(BaseModel):
    articles: List[FactExtractionArticle]

@app.post("/extract-facts")
def extract_facts(data: FactExtractionData):
    """
    Endpoint: /extract-facts
    ------------------------
    Purpose: Extract facts, statistics, quotes, or stories from given articles.

    Input (JSON body):
    {
      "articles": [
        {
          "title": "...",
          "full_content": "Full text of the article..."
        },
        ...
      ]
    }

    Output (JSON):
    {
      "analysis": [
        {
          "title": "...",
          "extracted_facts": [
             "Fact/Quote #1 (placeholder) ...",
             "Fact/Quote #2 (placeholder) ..."
          ]
        },
        ...
      ]
    }

    Note: This is a placeholder logic. In a real system, you might integrate:
    - Regex-based extraction,
    - A local language model to parse out facts/quotes,
    - An external NLP API, etc.
    """
    results = []
    for article in data.articles:
        # In a real application, you'd parse article.full_content with NLP logic.
        # For demonstration, we just return a placeholder list of "extracted" info.
        extracted_facts = [
            "Placeholder fact #1 from the article.",
            "Placeholder quote #2 from the article.",
        ]
        results.append({
            "title": article.title,
            "extracted_facts": extracted_facts
        })

    return {"analysis": results}
