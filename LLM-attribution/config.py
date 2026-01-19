
BASELINE_MODELS = {
    'TF-IDF': 'TF-IDF',
    'BERT': 'bert-base-uncased',
    'ELECTRA': 'google/electra-base-discriminator',
    'DeBERTa': 'microsoft/deberta-base'
}

EMBED_TYPES = {
    'TF-IDF': 'tf-idf',
    'BERT': 'bert',
    'ELECTRA': 'electra',
    'DeBERTa': 'deberta'
}

EVALUATION_COLUMNS = ['Prompt', 'Model', 'Accuracy', 'Weighted F1', 'Micro F1', 'Macro F1', 'Unsure']

DEFAULT_N_EVAL = 10
DEFAULT_REPS = 3