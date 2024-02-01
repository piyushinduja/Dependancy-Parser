from typing import List, Set
import torch.nn as nn
import torch
from torchtext.vocab import GloVe
from tqdm import tqdm
import numpy as np

POS_PATH = './data/pos_set.txt'
TRAIN_DATA = './data/train.txt'
DEV_DATA = './data/dev.txt'
TEST_DATA = './data/test.txt'
HIDDEN_DATA = './data/hidden.txt'
ACTIONS_PATH = './data/tagset.txt'
MODELS_PATH = './models'

WORD_EMBED_DIM = 300
GLOVE_NAME = '840B'
TYPE = 'cat'
EPOCHS = 20
LEARNING_RATE = 0.0001
torch.manual_seed(42)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
glove = GloVe(name=GLOVE_NAME, dim=WORD_EMBED_DIM)


class MulticlassClassifier(nn.Module):
    def __init__(self, e_dim) -> None:
        super().__init__()
        self.pos_embed = nn.Embedding(18, 50)
        self.word_lin = nn.Linear(e_dim, 200)
        if TYPE=='mean':
            self.pos_lin = nn.Linear(50, 200)
        else:
            self.pos_lin = nn.Linear(50*4, 200)
        self.relu = nn.ReLU()
        self.output = nn.Linear(200, 75)

    def forward(self, embeddings, pos):
        if TYPE=='mean':
            pos_embeds = self.pos_embed(pos).mean(1).squeeze(1)
            pos_embeds = torch.mean(pos_embeds, axis=0)
        else:
            pos_embeds = self.pos_embed(pos)
            pos_embeds = pos_embeds.view(pos_embeds.shape[0], pos_embeds.shape[1]*pos_embeds.shape[2])
        lin1 = self.word_lin(embeddings)
        lin2 = self.pos_lin(pos_embeds)
        lin = torch.add(lin1, lin2)
        relu = self.relu(lin)
        out = self.output(relu)
        return out

class Token:
    def __init__(self, idx: int, word: str, pos: str):
        self.idx = idx  # Unique index of the token
        self.word = word  # Token string
        self.pos = pos  # Part of speech tag


class DependencyEdge:
    def __init__(self, source: Token, target: Token, label: str):
        self.source = source  # Source token index
        self.target = target  # target token index
        self.label = label  # dependency label
        pass


class ParseState:
    def __init__(self, stack: List[Token], parse_buffer: List[Token], dependencies: List[DependencyEdge]):
        # A stack of token indices in the sentence. Assumption: the root token has index 0, the rest of the tokens in the sentence starts with 1.
        self.stack = stack
        self.parse_buffer = parse_buffer  # A buffer of token indices
        self.dependencies = dependencies
        pass

    def add_dependency(self, source_token, target_token, label):
        self.dependencies.append(
            DependencyEdge(
                source=source_token,
                target=target_token,
                label=label,
            )
        )


def shift(state: ParseState) -> None:
    # print("buffer[0] ", state.parse_buffer[0].pos)
    state.stack.append(state.parse_buffer[0])
    state.parse_buffer.pop(0)


def left_arc(state: ParseState, label: str) -> None:
    state.add_dependency(state.stack[-1], state.stack[-2], label)
    state.stack.pop(-2)


def right_arc(state: ParseState, label: str) -> None:
    state.add_dependency(state.stack[-2], state.stack[-1], label)
    state.stack.pop(-1)



def is_final_state(state: ParseState, cwindow: int) -> bool:
    if(len(state.parse_buffer) == 2 and len(state.stack)==3):
        return True
    return False

def get_tagset2ix(path):
    tagset2ix = {}
    count = 0
    with open(path) as f:
        data = f.readlines()
        for line in data:
            tagset2ix[line.strip()] = count
            count += 1
    return tagset2ix
pos_tag2ix = get_tagset2ix(POS_PATH)
actions_tag2ix = get_tagset2ix(ACTIONS_PATH)

def get_i_to_x(path):
    i_to_x = {}
    count = 0
    with open(path) as f:
        data = f.readlines()
        for line in data:
            i_to_x[count] = line.strip()
            count += 1
    return i_to_x
pos_i_to_x = get_i_to_x(POS_PATH)
actions_i_to_x = get_i_to_x(ACTIONS_PATH)

def evaluate_files(dev_lines, model):
        all_pred = []
        # gold = []
        words_list = []
        for line_no in tqdm(range(len(dev_lines))):
            sent_preds = []
            line = dev_lines[line_no]
            dev_words = line.split('|||')[0].strip().split(" ")
            words_list.append(dev_words)
            dev_pos = line.split('|||')[1].strip().split(" ")

            stack = [Token(-2, '[PAD]', 'NULL'), Token(-1, '[PAD]', 'NULL')]
            buffer = []
            for i in range(len(dev_words)):
                buffer.append(Token(i, dev_words[i], dev_pos[i]))
            buffer.append(Token(len(buffer), '[PAD]', 'NULL'))
            buffer.append(Token(len(buffer), '[PAD]', 'NULL'))
            parse_state = ParseState(stack=stack, parse_buffer=buffer, dependencies=[])

            while not is_final_state(parse_state,2):
                t1 = parse_state.stack[len(stack)-1]
                t2 = parse_state.stack[len(stack)-2]
                t3 = parse_state.parse_buffer[0]
                t4 = parse_state.parse_buffer[1]
                word_toks = [t2.word, t1.word, t3.word, t4.word]
                pos_toks = [t2.pos, t1.pos, t3.pos, t4.pos]

                word_embed = torch.ones(1, 4*WORD_EMBED_DIM)
                row_embed = []
                for i in word_toks:
                    row_embed.append(glove[i])
                word_embed = torch.cat((row_embed[0], row_embed[1], row_embed[2], row_embed[3]), 0)
                
                dev_pos_data = []
                for m in pos_toks:
                    dev_pos_data.append(pos_tag2ix[m])
                dev_pos_data = torch.tensor(dev_pos_data)
                x_dev = torch.cat((word_embed, dev_pos_data), 0)
                with torch.no_grad():
                    pred = model(x_dev[0:WORD_EMBED_DIM*4].view(1, x_dev[0:WORD_EMBED_DIM*4].shape[0]).to(device), x_dev[WORD_EMBED_DIM*4:(WORD_EMBED_DIM*4)+4].view(1, x_dev[WORD_EMBED_DIM*4:(WORD_EMBED_DIM*4)+4].shape[0]).to(torch.int64).to(device))
                  
                    actions, action_idx = torch.topk(pred, 75)
                    actions = actions.tolist()[0]
                    action_idx = action_idx.tolist()[0]
                    pred_action = actions_i_to_x[action_idx[0]]

                    # Handling Illegal Actions
                    if 'REDUCE' in pred_action:
                        if(len(parse_state.stack) < 3):
                            pred_action = 'SHIFT'
                    c = 0
                    if pred_action == "SHIFT":
                        if(len(parse_state.parse_buffer) == 2):
                            while pred_action == "SHIFT":
                                c = c + 1
                                pred_action = actions_i_to_x[action_idx[c]]

                    # Updating the Parse state
                    if pred_action == 'SHIFT':
                        shift(parse_state)
                    elif "REDUCE_L" in pred_action:
                        left_arc(parse_state, pred_action)
                    else:
                        right_arc(parse_state, pred_action)
                    
                    sent_preds.append(pred_action)
            all_pred.append(sent_preds)  

        return all_pred




from torch.nn.parallel.data_parallel import data_parallel
model_path = "./models/dev/10"
checkpoint = torch.load(model_path)

model = MulticlassClassifier(e_dim=1200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.load_state_dict(checkpoint["model_param"])
optimizer.load_state_dict(checkpoint["optim_param"])

# Loading and creating data
data = open('./data/hidden.txt', 'r', encoding='utf8')
data = list(data)

pred_actions = evaluate_files(data, model)

with open('./models/predicted4.txt', 'a', encoding='utf8') as f:
  for i in pred_actions:
    str1 = ""
    for j in range(len(i)):
      print(str1)
      if j==len(i)-1:
        str1 += i[j]
      else:  
        str1 += i[j] + " "
    f.write(f'{str1}\n')