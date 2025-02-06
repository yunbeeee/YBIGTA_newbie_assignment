import torch
from torch import nn, optim, Tensor, LongTensor, FloatTensor
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

from tqdm import tqdm
from sklearn.metrics import f1_score

from word2vec import Word2Vec
from model import MyGRULanguageModel
from config import *


if __name__ == "__main__":
    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # load Word2Vec checkpoint and get trained embeddings
    word2vec = Word2Vec(vocab_size, d_model, window_size, method)
    checkpoint = torch.load("word2vec.pt")
    word2vec.load_state_dict(checkpoint)
    embeddings = word2vec.embeddings_weight()

    # declare model, criterion and optimizer
    model = MyGRULanguageModel(d_model, hidden_size, num_classes, embeddings).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # load train, validation dataset
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=True)

    # train
    for epoch in tqdm(range(num_epochs)):
        loss_sum: float = 0.0
        train_preds: list[int] = []
        train_labels: list[int] = []
        for data in train_loader:
            optimizer.zero_grad()
            train_input_ids: Tensor = tokenizer(data["verse_text"], padding=True, return_tensors="pt")\
                .input_ids.to(device)
            train_labels_tensor: Tensor = torch.as_tensor(data["label"], dtype=torch.long, device=device).squeeze()
          
            train_logits: Tensor = model(train_input_ids)

            loss: Tensor = criterion(train_logits, train_labels_tensor)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        preds: list[int] = []
        labels_list: list[int] = []
        with torch.no_grad():
            for data in validation_loader:
                input_ids: Tensor = tokenizer(data["verse_text"], padding=True, return_tensors="pt")\
                    .input_ids.to(device)
                logits: Tensor = model(input_ids)

                labels_list.extend([int(label) for label in data["label"]])

                # logits 변환 (다차원 리스트 방지)
                batch_preds = logits.argmax(-1).squeeze().cpu().tolist()
                if isinstance(batch_preds, list):
                    preds.extend(batch_preds)
                else:
                    preds.append(int(batch_preds))

        # 리스트 중첩 여부 확인 후 펼치기
        # preds = [item for sublist in preds for item in sublist] if any(isinstance(i, list) for i in preds) else preds
        # labels_list = [item for sublist in labels for item in sublist] if any(isinstance(i, list) for i in labels) else labels
        labels_list = [int(label) for label in labels_list]  # 리스트 변환
        preds = [int(pred) for pred in preds]  # 리스트 변환
        
        macro = f1_score(labels_list, preds, average='macro')
        micro = f1_score(labels_list, preds, average='micro')
        print(f"loss: {loss_sum/len(train_loader):.6f} | macro: {macro:.6f} | micro: {micro:.6f}")

    # save model checkpoint
    torch.save(model.cpu().state_dict(), "checkpoint.pt")