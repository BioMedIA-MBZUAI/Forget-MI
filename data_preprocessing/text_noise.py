import random 
import numpy as np 
import pandas as pd 
import re

def extract_words(sentence, synonyms_df):

    # Extracts words from the sentence that are present in the 'second_word' column. If more than two words are found, keeps only two at random.
    words_in_sentence = sentence.split()
    extracted_words = [word for word in words_in_sentence if word in synonyms_df['second_word'].values]

    # If more than 2 words are extracted, select only 2 at random
    if len(extracted_words) > 1:
        extracted_words = random.sample(extracted_words, 1)
    
    return extracted_words

def find_first_word(synonyms_df, word):

    # Finds the first_word corresponding to a given second_word based on the highest similarity.
    filtered_rows = synonyms_df[synonyms_df['second_word'] == word]
    highest_similarity_row = filtered_rows.loc[filtered_rows['similarity'].idxmax()]
    return highest_similarity_row['first_word']

def change_synonym(sentence, synonyms_df):
    extracted_words = extract_words(sentence, synonyms_df)
    results = {word: find_first_word(synonyms_df, word) for word in extracted_words}
    modified_sentence = " ".join([results.get(word, word) for word in sentence.split()])

    return modified_sentence

def add_noise_char(text, noise_level=0.1):
    words = text.split()
    noisy_words = []
    for word in words:
        if random.random() < noise_level:
            noise_type = random.choice(["insert", "delete", "substitute", "swap"])
            noisy_word = word
            if noise_type == "insert":
                pos = random.randint(0, len(word))
                noisy_word = word[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[pos:]
            elif noise_type == "delete" and len(word) > 1:
                pos = random.randint(0, len(word) - 1)
                noisy_word = word[:pos] + word[pos + 1:]
            elif noise_type == "substitute":
                pos = random.randint(0, len(word) - 1)
                noisy_word = word[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[pos + 1:]
            elif noise_type == "swap" and len(word) > 1:
                pos = random.randint(0, len(word) - 2)
                noisy_word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
            noisy_words.append(noisy_word)
        else:
            noisy_words.append(word)
    return " ".join(noisy_words)


def add_noise_word(sentence, synonyms_df, noise_level=0.1):

    extracted_words = extract_words(sentence, synonyms_df)

    if len(extracted_words) > 0: 
        results = {word: find_first_word(synonyms_df, word) for word in extracted_words}
        sentence = " ".join([results.get(word, word) for word in sentence.split()])

    words = sentence.split()
    noisy_words = []
    
    for i, word in enumerate(words):
        if random.random() < noise_level:
            noise_type = random.choice(["delete", "repeat", "shuffle"])
            
            if noise_type == "delete":
                continue
            
            elif noise_type == "repeat":
                noisy_words.append(word)
                noisy_words.append(word)
                
            elif noise_type == "shuffle" and len(words) > 1:
                swap_idx = random.randint(0, len(words) - 1)
                words[i], words[swap_idx] = words[swap_idx], words[i]
        
        noisy_words.append(word)
    
    return " ".join(noisy_words)


def add_noise(text, noise_level, synonyms_df):
    text = remove_keywords(text)
    if random.choice(["word", "char"]) == "word":
        return add_noise_word(text, synonyms_df, noise_level)
    else:
        return add_noise_char(text, noise_level)
    
def remove_keywords(text, max_removals=2):
    """
    Remove up to a maximum of two keywords from the text if they exist.
    
    Args:
        text (str): The input text.
        keywords (list): List of keywords to check for.
        max_removals (int): Maximum number of keywords to remove.
    
    Returns:
        str: Modified text with up to `max_removals` keywords removed.
    """
    #Top 10 Keywords: [('atelectasis', 5), ('lung', 5), ('edema', 4), ('cardiomegaly', 3), ('silhouette', 3), ('pneumothorax', 3), ('markings', 3), ('disease', 3), ('NUMBER', 3), ('chest', 2)]
    keywords = ["atelectasis", "lung", "edema", "cardiomegaly", "silhouette", "pneumothorax", "markings", "disease", "NUMBER", "chest"]
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