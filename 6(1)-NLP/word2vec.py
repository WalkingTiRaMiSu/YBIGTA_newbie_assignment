import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        #pass

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        # 구현하세요!
        #pass
        
        # 1. 불용어 사전 구축
        # stop_words = ["the", "a", "an", "and", "of", "to", "in", "for", "on", "with", "at", "from"]
        # stop_token_ids = {
        #     tid for word in stop_words
        #     for tid in tokenizer.encode(word, add_special_tokens=False)
        # }

        # 2. 문장 길이 필터링 (길이가 충분한 문장만 사용)
        filtered_corpus = [
            text for text in corpus
            if len(tokenizer.encode(text, add_special_tokens=False)) >= 0
        ]

        # 3. 토큰화 + 불용어 제거
        # encoded = [
        #     tokenizer.encode(text, add_special_tokens=False)
        #     for text in filtered_corpus
        # ]
        # token_ids = [
        #     tid for sent in encoded for tid in sent
        #     if tid != tokenizer.pad_token_id and tid not in stop_token_ids
        # ]

        encoded = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in filtered_corpus
        ]
        token_ids = [
            tid for sent in encoded for tid in sent
            if tid != tokenizer.pad_token_id
        ]
        
        max_tokens = 10000  
        token_ids = token_ids[:max_tokens]

        print(f"Total tokens: {len(token_ids)}")
        print("tokenizer vocab size:", tokenizer.vocab_size)
        print(f"Effective training steps: {len(token_ids) - 2 * self.window_size}")

        for epoch in range(num_epochs):
            total_loss: float = 0.0
            if self.method == "cbow":
                total_loss = self._train_cbow(token_ids, criterion, optimizer)
            elif self.method == "skipgram":
                total_loss = self._train_skipgram(token_ids, criterion, optimizer)

            print(f"Epoch {epoch+1}/{num_epochs} Loss: {total_loss:.4f}")

    def _train_cbow(self, token_ids, criterion, optimizer) -> float:
    #     self,
    #     # 구현하세요!
    # ) -> None:
    #     # 구현하세요!
    #     pass
        self.train()
        total_loss: float = 0.0
        contexts, targets = [], []
        batch_size = 64

        total_steps = len(token_ids) - 2 * self.window_size
        current_step = 0
        print_every = 1000

        for center_idx in range(self.window_size, len(token_ids) - self.window_size):
            context = token_ids[center_idx - self.window_size:center_idx] + token_ids[center_idx+1:center_idx + self.window_size + 1]
            target = token_ids[center_idx]

            contexts.append(context)
            targets.append(target)
            current_step += 1

            if len(contexts) == batch_size:
                # Tensor화
                context_tensor = torch.tensor(contexts, device=self.embeddings.weight.device)  # (B, 2w)
                target_tensor = torch.tensor(targets, device=self.embeddings.weight.device)     # (B,)

                context_embeds = self.embeddings(context_tensor)  # (B, 2w, D)
                context_mean = context_embeds.mean(dim=1)         # (B, D)

                logits = self.weight(context_mean)                # (B, V)
                loss = criterion(logits, target_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                contexts, targets = [], []

                if current_step % print_every < 2:
                    print(f"[CBOW] Step {current_step}/{total_steps}  Loss: {loss.item():.4f}")

        # 남은 배치 처리
        if contexts:
            context_tensor = torch.tensor(contexts, device=self.embeddings.weight.device)
            target_tensor = torch.tensor(targets, device=self.embeddings.weight.device)

            context_embeds = self.embeddings(context_tensor)
            context_mean = context_embeds.mean(dim=1)
            logits = self.weight(context_mean)
            loss = criterion(logits, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"[CBOW] Final Step {current_step}/{total_steps}  Loss: {loss.item():.4f}")
        result:float = total_loss / total_steps

        return result

    def _train_skipgram(self, token_ids, criterion, optimizer)-> float:
    #     self,
    #     # 구현하세요!
    # ) -> None:
    #     # 구현하세요!
    #     pass
        self.train()
        total_loss: float = 0.0
        centers, targets = [], []
        batch_size = 64  # 💡 너가 조절 가능!

        for center_idx in range(self.window_size, len(token_ids) - self.window_size):
            center = token_ids[center_idx]
            context = token_ids[center_idx - self.window_size:center_idx] + token_ids[center_idx+1:center_idx + self.window_size + 1]

            for target in context:
                centers.append(center)
                targets.append(target)

                if len(centers) == batch_size:
                    center_tensor = torch.tensor(centers, device=self.embeddings.weight.device)
                    target_tensor = torch.tensor(targets, device=self.embeddings.weight.device)

                    center_embeds = self.embeddings(center_tensor)  # (B, D)
                    logits = self.weight(center_embeds)             # (B, V)

                    loss = criterion(logits, target_tensor)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    centers, targets = [], []  # 💡 초기화

        # 남은 미니배치 처리
        if centers:
            center_tensor = torch.tensor(centers, device=self.embeddings.weight.device)
            target_tensor = torch.tensor(targets, device=self.embeddings.weight.device)
            center_embeds = self.embeddings(center_tensor)
            logits = self.weight(center_embeds)
            loss = criterion(logits, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 평균 loss 반환
        total_num_context_pairs = (len(token_ids) - 2 * self.window_size) * (2 * self.window_size)
        num_batches = total_num_context_pairs // batch_size
        result:float = total_loss / max(1, num_batches)
        return result
