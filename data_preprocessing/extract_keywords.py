import pandas as pd
import random
import re
import spacy
from collections import Counter

# Load the TSV file
df = pd.read_csv("./metadata/all_data.tsv", sep="\t", header=None)

all_text = " ".join(df[4].astype(str).tolist())

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

chunk_size = 100000  # Define the chunk size (number of characters)
chunks = [all_text[i:i + chunk_size] for i in range(0, len(all_text), chunk_size)]

# Process each chunk individually
docs = [nlp(chunk) for chunk in chunks]

# Combine the results (e.g., extracting tokens or keywords)
all_keywords = []
for doc in docs:
    keywords = [token.text for token in doc if token.pos_ in {"NOUN", "PROPN"}]
    all_keywords.extend(keywords)

# # Process the combined text
# doc = nlp(all_text)

# # Extract keywords: Nouns and Proper Nouns
# keywords = [token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop]
print("Number of Keywords:", len(keywords))

# Get the most common keywords
top_keywords = Counter(keywords).most_common(10)

# Display top 10 keywords
print("Top 10 Keywords:", top_keywords)

# Define the method for adding noise
def add_limited_noise(text, keywords, max_removals=2):
    """
    Remove up to a maximum of two keywords from the text if they exist.
    
    Args:
        text (str): The input text.
        keywords (list): List of keywords to check for.
        max_removals (int): Maximum number of keywords to remove.
    
    Returns:
        str: Modified text with up to `max_removals` keywords removed.
    """
    # Check which keywords exist in the text
    existing_keywords = [word for word, _ in keywords if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)]
    print("\n Existing Keywords:", existing_keywords)
    
    # If no keywords are found, return the text as-is
    if not existing_keywords:
        return text
    
    # Randomly select up to `max_removals` keywords to remove
    keywords_to_remove = random.sample(existing_keywords, min(len(existing_keywords), max_removals))
    print("\n Keywords to Remove:", keywords_to_remove)
    
    # Create a regex pattern for the selected keywords
    pattern = r"\b(" + "|".join(re.escape(word) for word in keywords_to_remove) + r")\b"
    
    # Replace the selected keywords with a placeholder or remove them
    modified_text = re.sub(pattern, "[NOISE]", text, flags=re.IGNORECASE)
    return modified_text

# text = "The heart size is moderately enlarged similar to prior study with the cephalization of the pulmonary vasculature and minimal increased reticulation suggestive of minimal interstitial edema . The lungs are otherwise clear without focal consolidation . There is no pleural effusion or pneumothorax . The osseous structures are locally demineralized with prominent kyphotic angulation of the thoracic spine . Moderate cardiomegaly with mild fluid overload and minimal interstitial edema improved from prior study"
# noisy_text = add_limited_noise(text, top_keywords)
# print("\n Original Text:", text)
# print("\n Noisy Text:", noisy_text)