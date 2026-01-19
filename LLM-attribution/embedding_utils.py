import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer
from typing import List


class EmbeddingGenerator:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def embed_texts(
            self,
            texts: List[str],
            model_name: str,
            baseline_type: str = 'bert',
            batch_size: int = 16
    ) -> torch.Tensor:
        if baseline_type in ['bert', 'deberta', 'electra']:
            return self._embed_with_transformers(texts, model_name, baseline_type, batch_size)
        elif baseline_type == 'tf-idf':
            return self._embed_with_tfidf(texts)
        else:
            raise ValueError(f"不支持的baseline_type: {baseline_type}")

    def _embed_with_transformers(
            self,
            texts: List[str],
            model_name: str,
            baseline_type: str,
            batch_size: int
    ) -> torch.Tensor:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(self.device)
        model.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                tokenized_texts = tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )

                input_ids = tokenized_texts.input_ids.to(self.device)
                attention_mask = tokenized_texts.attention_mask.to(self.device)

                outputs = model(input_ids, attention_mask)
                batch_embedding = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(batch_embedding.cpu())

        if len(all_embeddings) > 0:
            embedding = torch.cat(all_embeddings, dim=0)
        else:
            embedding = torch.tensor([])

        return embedding

    def _embed_with_tfidf(self, texts: List[str]) -> torch.Tensor:
        texts=[texts] if not isinstance(texts, str) else list(texts)
        vectorizer = TfidfVectorizer(
            max_features=3000,
            analyzer='char',
            ngram_range=(4, 4)
        )
        embedding = torch.from_numpy(vectorizer.fit_transform(texts).toarray())
        return embedding

    @staticmethod
    def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(embeddings)