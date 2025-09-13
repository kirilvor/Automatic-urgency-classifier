import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Dummy data pro trénink (příklady zpráv z datových schránek)
texts = ["urgent payment required", "information update", "immediate action needed", "monthly report"]
labels = [1, 0, 1, 0]  # 1 = urgentní, 0 = neurgentní

# Jednoduchá vektorizace (bag-of-words)
vocab = list(set(" ".join(texts).split()))
word_to_ix = {word: i for i, word in enumerate(vocab)}

X = torch.stack([vectorize(t) for t in texts])
Y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

def vectorize(text):
    vec = torch.zeros(len(vocab))
    for word in text.split():
        if word in word_to_ix:
            vec[word_to_ix[word]] += 1
    return vec

# Model: Jednoduchá neuronová síť pro klasifikaci
class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

model = Classifier(len(vocab))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Trénink
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=2)
for epoch in range(100):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()

test_text = "urgent action required"
test_vec = vectorize(test_text)
pred = model(test_vec)
print(f"Prediction for '{test_text}': {pred.item()} (closer to 1 means urgent)")

test_text2 = "regular update"
test_vec2 = vectorize(test_text2)
pred2 = model(test_vec2)
print(f"Prediction for '{test_text2}': {pred2.item()} (closer to 0 means non-urgent)")
