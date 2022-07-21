import torch
import json

from model import Glove
from vectorizer import GloveDataset

from tqdm import tqdm
from torch.utils.data import DataLoader



N_EMBEDDING = 100
BATCH_SIZE = 512
NUM_EPOCH = 5   
MIN_WORD_OCCURENCES = 10
X_MAX = 100
ALPHA = 0.75
BETA = 1e-8
RIGHT_WINDOW = 10

with open('/data/giturra/1e5tweets.txt', encoding='utf-8') as reader:
    corpus = [line for line in reader]

device = torch.device('cuda:2')

dataset = GloveDataset(corpus, right_window=RIGHT_WINDOW, min_count=MIN_WORD_OCCURENCES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Glove(dataset.indexer.n_words, N_EMBEDDING, x_max=X_MAX, alpha=ALPHA)
model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-8)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=BETA)

for _ in tqdm(range(NUM_EPOCH)):
    for batch in tqdm(dataloader):
        i = batch[0].to(device)
        j = batch[1].to(device)
        w = batch[2].to(device)
        optimizer.zero_grad()
        loss = model(i, j, w)
        # print(loss.item())
        #avg_loss += loss.item() / num_batches
        loss.backward()
        optimizer.step()



torch.save(model.state_dict(), './model/glove_model.pt')
with open('./data/vocab.json', "w", encoding='utf-8') as vocab_file:
    json.dump(
        {'vocab':dataset.indexer.word_to_index, 'emb_size':N_EMBEDDING}, vocab_file
    )
