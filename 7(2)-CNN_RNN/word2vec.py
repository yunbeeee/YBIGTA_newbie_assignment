import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

import random
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
        self.embeddings = nn.Embedding(vocab_size, d_model, padding_idx=0)
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
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!
        # 🔹 빈 문장 방지: tokenized_corpus를 생성할 때 필터링 추가
        tokenized_corpus = []
        for sent in corpus:
            tokenized_sentence = tokenizer.encode(sent, add_special_tokens=False)
            if not tokenized_sentence:  # 빈 리스트인 경우 건너뛰기
                continue
            tokenized_corpus.append(tokenized_sentence)

        for epoch in range(num_epochs):
            total_loss: float = 0.0  # 🔹 불필요한 그래디언트 추적 방지
            for sentence in tokenized_corpus:
                if len(sentence) < 2:  # 🔹 너무 짧은 문장은 학습할 필요 없음
                    continue
                
                loss: Tensor
                if self.method == "cbow":
                    loss = self._train_cbow(sentence, criterion, optimizer)
                else:
                    loss = self._train_skipgram(sentence, criterion, optimizer)

                total_loss += loss.item()  

    def _train_cbow(
        self, sentence: list[int], criterion, optimizer
        # 구현하세요!
    ) -> Tensor:
        # 구현하세요!
        context_window = self.window_size // 2
        total_loss: Tensor = torch.tensor(0.0)

        for i in range(context_window, len(sentence) - context_window):
            context = sentence[i - context_window:i] + sentence[i + 1:i + context_window + 1]
            target = sentence[i]

            if target == 0 or len(context) == 0:
                continue
            
            context_embeds = self.embeddings(LongTensor(context)).mean(dim=0)  # 평균 임베딩
            logits = self.weight(context_embeds)
            loss = criterion(logits.unsqueeze(0), LongTensor([target]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()

        return total_loss

    def _train_skipgram(
        self, sentence: list[int], criterion, optimizer
        # 구현하세요!
    ) -> Tensor:
        # 구현하세요!
        context_window = self.window_size // 2
        total_loss: Tensor = torch.tensor(0.0)

        for i in range(context_window, len(sentence) - context_window):
            target = sentence[i]
            context = sentence[i - context_window:i] + sentence[i + 1:i + context_window + 1]
            
            if target == 0 or len(context) == 0:
                continue


            target_embed = self.embeddings(LongTensor([target]))
            logits = self.weight(target_embed).squeeze(0)

            #context_tensor = torch.tensor(context, dtype=torch.long)
            random_target = random.choice(context)
            target_tensor = torch.tensor([random_target], dtype=torch.long)
            loss = criterion(logits.unsqueeze(0), target_tensor)

            #loss = criterion(logits, LongTensor(context))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.detach()

        return total_loss

    # 구현하세요!
    #pass