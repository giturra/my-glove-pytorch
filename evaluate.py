import json
import torch
from web.datasets.similarity import fetch_MEN
from web.evaluate import evaluate_similarity

from model import Glove

with open('./data/vocab.json', encoding='utf-8') as reader:
    data = json.load(reader)

vocab = data['vocab']
emb_size = data['emb_size']

PATH = './model/glove_model.pt'

device = torch.device('cpu')
model = Glove(len(vocab.keys()), emb_size)
model.load_state_dict(torch.load(PATH, map_location=device))

MEN = fetch_MEN()
embs = {}

for word, idx in vocab.items():
    embs[word] = (model.emb_u.weight[idx] + model.emb_v.weight[idx]).detach().cpu().numpy()

print(evaluate_similarity(embs, MEN.X, MEN.y))