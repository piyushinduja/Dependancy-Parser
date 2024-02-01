from typing import List, Set
import torch.nn as nn
import torch
from collections import deque
from torch.utils.data import TensorDataset, DataLoader
import random
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
        # pos_embeds = self.pos_embed(pos).mean(1).squeeze(1)
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


def new_state(sent, pos, dep):
    words = sent.split(" ")
    words.append('[PAD]')
    words.append('[PAD]')
    pos = pos.split(" ")
    pos.append('NULL')
    pos.append('NULL')
    buffer = []
    for i in range(words.__len__()):
        new_token = Token(i, words[i], pos[i])
        buffer.append(new_token)
    buffer = deque(buffer)
    dep = dep.split(" ")
    stack = [Token(-2, '[PAD]', 'NULL'), Token(-1, '[PAD]', 'NULL')]
    parse_state = [[[stack[len(stack) - 2], stack[len(stack) - 1], buffer[0], buffer[1]], dep[0]]]
    for i in range(1, len(dep)):
        if 'SHIFT' in dep[i - 1]:
            stack.append(buffer[0])
            buffer.popleft()
        elif 'REDUCE_L' in dep[i - 1]:
            w1 = stack.pop()
            stack.pop()
            stack.append(w1)
        elif 'REDUCE_R' in dep[i - 1]:
            stack.pop()
            w1 = stack.pop()
            stack.append(w1)
        parse_state.append([[stack[len(stack) - 2], stack[len(stack) - 1], buffer[0], buffer[1]], dep[i]])

    return parse_state


def break_tokens(state):
    words = []
    pos = []
    for i in range(4):
        words.append(state[0][i].word)
        pos.append(state[0][i].pos)
    return [words, pos, state[1]]


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


def make_glove_embed(word_tokens):
    # glove = GloVe(name=GLOVE_NAME, dim=WORD_EMBED_DIM)
    word_embed = torch.ones(len(word_tokens), WORD_EMBED_DIM)
    if TYPE=='mean':
        word_embed = torch.ones(len(word_tokens), WORD_EMBED_DIM)
        for i in range(len(word_tokens)):
            words = word_tokens[i]
            row_embed = []
            for word in words:
                row_embed.append(glove[word].tolist())
            row_embed = torch.tensor(row_embed)
            word_embed[i] = torch.mean(row_embed, 0)
    else:
        word_embed = torch.ones(len(word_tokens), 4*WORD_EMBED_DIM)
        for i in range(len(word_tokens)):
            words = word_tokens[i]
            row_embed = []
            for word in words:
                    row_embed.append(glove[word])
            word_embed[i] = torch.cat((row_embed[0], row_embed[1], row_embed[2], row_embed[3]), 0)
    return word_embed


def create_data(path):
    data = open(path, 'r', encoding='utf8')
    data = list(data)
    sentences = [i.split('|||')[0].strip() for i in data]
    pos = [i.split('|||')[1].strip() for i in data]
    dep = [i.split('|||')[2].strip() for i in data]

    sent_states = []
    for i in range(sentences.__len__()):
        sent_states.append(new_state(sentences[i], pos[i], dep[i]))
    
    all_states = []
    for i in sent_states:
        for j in i:
            all_states.append(j)

    word_tokens = []
    pos_tokens = []
    dep_tokens = []
    for i in all_states:
        l1 = break_tokens(i)
        word_tokens.append(l1[0])
        pos_tokens.append(l1[1])
        dep_tokens.append(l1[2])

    pos_data = []
    dep_data = []
    for i in range(len(pos_tokens)):
        x = []
        for j in pos_tokens[i]:
            x.append(pos_tag2ix[j])
        pos_data.append(x)
        dep_data.append(actions_tag2ix[dep_tokens[i]])

    pos_data = torch.tensor(pos_data)
    dep_data = torch.tensor(dep_data)

    word_embed = make_glove_embed(word_tokens)
        
    return (word_embed, pos_data, dep_data)


def get_deps(words_lists, actions, cwindow):
    
    all_deps = []   # List of List of dependencies
    # Iterate over sentences
    for w_ix, words_list in enumerate(words_lists):
        # Intialize stack and buffer appropriately
        stack = [Token(idx=-i-1, word="[NULL]", pos="NULL") for i in range(cwindow)]
        parser_buff = []
        for ix in range(len(words_list)):
            parser_buff.append(Token(idx=ix, word=words_list[ix], pos="NULL"))
        parser_buff.extend([Token(idx=ix+i+1, word="[NULL]",pos="NULL") for i in range(cwindow)])
        # Initilaze the parse state
        state = ParseState(stack=stack, parse_buffer=parser_buff, dependencies=[])

        # Iterate over the actions and do the necessary state changes
        for action in actions[w_ix]:
            if action == "SHIFT":
                shift(state)
            elif action[:8] == "REDUCE_L":
                left_arc(state, action[9:])
            else:
                right_arc(state, action[9:])
        assert is_final_state(state,cwindow)    # Check to see that the parse is complete
        right_arc(state, "root")    # Add te root dependency for the remaining element on stack
        all_deps.append(state.dependencies.copy())  # Copy over the dependenices found
    return all_deps


def compute_metrics(words_lists, gold_actions, pred_actions, cwindow=2):
    lab_match = 0  # Counter for computing correct head assignment and dep label
    unlab_match = 0 # Counter for computing correct head assignments
    total = 0       # Total tokens

    # Get all the dependencies for all the sentences
    gold_deps = get_deps(words_lists, gold_actions, cwindow)    # Dep according to gold actions
    pred_deps = get_deps(words_lists, pred_actions, cwindow)    # Dep according to predicted actions

    # Iterate over sentences
    for w_ix, words_list in enumerate(words_lists):
        # Iterate over words in a sentence
        for ix, word in enumerate(words_list):
            # Check what is the head of the word in the gold dependencies and its label
            for dep in gold_deps[w_ix]:
                if dep.target.idx == ix:
                    gold_head_ix = dep.source.idx
                    gold_label = dep.label
                    break
            # Check what is the head of the word in the predicted dependencies and its label
            for dep in pred_deps[w_ix]:
                if dep.target.idx == ix:
                    # Do the gold and predicted head match?
                    if dep.source.idx == gold_head_ix:
                        unlab_match += 1
                        # Does the label match? 
                        if dep.label == gold_label:
                            lab_match += 1
                    break
            total += 1

    return unlab_match/total, lab_match/total
    

def __main__():

    train_data = create_data(TRAIN_DATA)
    x_data = torch.cat((train_data[0], train_data[1]), 1)
    y_data = train_data[2]
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    dev_lines = open(DEV_DATA, 'r', encoding='utf8')
    dev_lines = list(dev_lines)

    test_lines = open(TEST_DATA, 'r', encoding='utf8')
    test_lines = list(test_lines)

    if TYPE=='mean':
        model = MulticlassClassifier(e_dim=WORD_EMBED_DIM).to(device)
    else:
        model = MulticlassClassifier(e_dim=WORD_EMBED_DIM*4).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_dict = {"epoch": 0, "las": 0, "uas": 0, "model": {}, "optimizer": {}}

    for epoch in range(EPOCHS):
        print('Epoch: ', epoch+1)
        old_las = 0
        train_losses = []
        for x, y in tqdm(dataloader):
            model.train()
            if(TYPE=='mean'):
                pred = model(x[:, :WORD_EMBED_DIM].to(device), x[:, WORD_EMBED_DIM:WORD_EMBED_DIM+4].to(torch.int64).to(device))
            else:
                pred = model(x[:, :WORD_EMBED_DIM*4].to(device), x[:, WORD_EMBED_DIM*4:(WORD_EMBED_DIM*4)+4].to(torch.int64).to(device))
            loss = loss_func(pred, y.view(y.shape[0]).long().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print(f'Training Loss {np.average(train_losses)}')

    
        all_pred = []
        gold = []
        words_list = []
        for line_no in tqdm(range(len(dev_lines))):
            sent_preds = []
            line = dev_lines[line_no]
            dev_words = line.split('|||')[0].strip().split(" ")
            # print(dev_words)
            words_list.append(dev_words)
            dev_pos = line.split('|||')[1].strip().split(" ")
            dev_deps = line.split('|||')[2].strip().split(" ")
            gold.append(dev_deps)

            stack = [Token(-2, '[PAD]', 'NULL'), Token(-1, '[PAD]', 'NULL')]
            buffer = []
            for i in range(len(dev_words)):
                buffer.append(Token(i, dev_words[i], dev_pos[i]))
            buffer.append(Token(len(buffer), '[PAD]', 'NULL'))
            buffer.append(Token(len(buffer), '[PAD]', 'NULL'))
            # buffer = deque(buffer)
            parse_state = ParseState(stack=stack, parse_buffer=buffer, dependencies=[])

            while not is_final_state(parse_state,2):
                t1 = parse_state.stack[len(stack)-1]
                t2 = parse_state.stack[len(stack)-2]
                t3 = parse_state.parse_buffer[0]
                t4 = parse_state.parse_buffer[1]
                word_toks = [t2.word, t1.word, t3.word, t4.word]
                pos_toks = [t2.pos, t1.pos, t3.pos, t4.pos]
                # print(pos_toks)

                word_embed = torch.ones(1, WORD_EMBED_DIM)
                if TYPE=='mean':
                    word_embed = torch.ones(1, WORD_EMBED_DIM)
                    row_embed = []
                    for i in word_toks:
                        row_embed.append(glove[i].tolist())
                    row_embed = torch.tensor(row_embed)
                    word_embed = torch.mean(row_embed, 0)
                else:
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
                    if(TYPE=='mean'):
                        pred = model(x_dev[0:WORD_EMBED_DIM].view(1, x_dev[0:WORD_EMBED_DIM].shape[0]).to(device), x_dev[WORD_EMBED_DIM:WORD_EMBED_DIM+4].view(1, x_dev[WORD_EMBED_DIM:(WORD_EMBED_DIM)+4].shape[0]).to(torch.int64).to(device))
                        # pred = model(x_dev[0:WORD_EMBED_DIM].view(1, x_dev[0:WORD_EMBED_DIM].shape[0]), x_dev[WORD_EMBED_DIM:WORD_EMBED_DIM+4].view(1, x_dev[WORD_EMBED_DIM:(WORD_EMBED_DIM)+4].shape[0]).to(torch.int64))
                    else:
                        pred = model(x_dev[0:WORD_EMBED_DIM*4].view(1, x_dev[0:WORD_EMBED_DIM*4].shape[0]).to(device), x_dev[WORD_EMBED_DIM*4:(WORD_EMBED_DIM*4)+4].view(1, x_dev[WORD_EMBED_DIM*4:(WORD_EMBED_DIM*4)+4].shape[0]).to(torch.int64).to(device))
                        # pred = model(x_dev[0:WORD_EMBED_DIM*4].view(1, x_dev[0:WORD_EMBED_DIM*4].shape[0]), x_dev[WORD_EMBED_DIM*4:(WORD_EMBED_DIM*4)+4].view(1, x_dev[WORD_EMBED_DIM*4:(WORD_EMBED_DIM*4)+4].shape[0]).to(torch.int64))
                    
                    actions, action_idx = torch.topk(pred, 75)
                    actions = actions.tolist()[0]
                    action_idx = action_idx.tolist()[0]
                    pred_action = actions_i_to_x[action_idx[0]]
                    # print("Pred action", pred_action)

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
        
        uas, las = compute_metrics(words_list, gold, all_pred, 2)
        if(las > best_dict['las']):
          best_dict['epoch'] = epoch + 1
          best_dict['las'] = las
          best_dict['uas'] = uas
          best_dict['model'] = model.state_dict()
          best_dict['optimizer'] = optimizer.state_dict()
          torch.save({
            "model_param": best_dict["model"],
            "optim_param": best_dict["optimizer"],
            "epoch": best_dict["epoch"],
            "las": best_dict["las"],
            "uas": best_dict["uas"]
        }, f"{MODELS_PATH}/dev/{epoch+1}")

        print(f"Dev UAS at epoch {epoch + 1}: ", uas)
        print(f"Dev LAS at epoch {epoch + 1}: ", las) 


        # Test Eval
        all_pred = []
        gold = []
        words_list = []
        for line_no in tqdm(range(len(test_lines))):
            sent_preds = []
            line = test_lines[line_no]
            dev_words = line.split('|||')[0].strip().split(" ")
            # print(dev_words)
            words_list.append(dev_words)
            dev_pos = line.split('|||')[1].strip().split(" ")
            dev_deps = line.split('|||')[2].strip().split(" ")
            gold.append(dev_deps)

            stack = [Token(-2, '[PAD]', 'NULL'), Token(-1, '[PAD]', 'NULL')]
            buffer = []
            for i in range(len(dev_words)):
                buffer.append(Token(i, dev_words[i], dev_pos[i]))
            buffer.append(Token(len(buffer), '[PAD]', 'NULL'))
            buffer.append(Token(len(buffer), '[PAD]', 'NULL'))
            # buffer = deque(buffer)
            parse_state = ParseState(stack=stack, parse_buffer=buffer, dependencies=[])

            while not is_final_state(parse_state,2):
                t1 = parse_state.stack[len(stack)-1]
                t2 = parse_state.stack[len(stack)-2]
                t3 = parse_state.parse_buffer[0]
                t4 = parse_state.parse_buffer[1]
                word_toks = [t2.word, t1.word, t3.word, t4.word]
                pos_toks = [t2.pos, t1.pos, t3.pos, t4.pos]
                # print(pos_toks)

                word_embed = torch.ones(1, WORD_EMBED_DIM)
                if TYPE=='mean':
                    word_embed = torch.ones(1, WORD_EMBED_DIM)
                    row_embed = []
                    for i in word_toks:
                        row_embed.append(glove[i].tolist())
                    row_embed = torch.tensor(row_embed)
                    word_embed = torch.mean(row_embed, 0)
                else:
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
                    if(TYPE=='mean'):
                        pred = model(x_dev[0:WORD_EMBED_DIM].view(1, x_dev[0:WORD_EMBED_DIM].shape[0]).to(device), x_dev[WORD_EMBED_DIM:WORD_EMBED_DIM+4].view(1, x_dev[WORD_EMBED_DIM:(WORD_EMBED_DIM)+4].shape[0]).to(torch.int64).to(device))
                        # pred = model(x_dev[0:WORD_EMBED_DIM].view(1, x_dev[0:WORD_EMBED_DIM].shape[0]), x_dev[WORD_EMBED_DIM:WORD_EMBED_DIM+4].view(1, x_dev[WORD_EMBED_DIM:(WORD_EMBED_DIM)+4].shape[0]).to(torch.int64))
                    else:
                        pred = model(x_dev[0:WORD_EMBED_DIM*4].view(1, x_dev[0:WORD_EMBED_DIM*4].shape[0]).to(device), x_dev[WORD_EMBED_DIM*4:(WORD_EMBED_DIM*4)+4].view(1, x_dev[WORD_EMBED_DIM*4:(WORD_EMBED_DIM*4)+4].shape[0]).to(torch.int64).to(device))
                        # pred = model(x_dev[0:WORD_EMBED_DIM*4].view(1, x_dev[0:WORD_EMBED_DIM*4].shape[0]), x_dev[WORD_EMBED_DIM*4:(WORD_EMBED_DIM*4)+4].view(1, x_dev[WORD_EMBED_DIM*4:(WORD_EMBED_DIM*4)+4].shape[0]).to(torch.int64))
                    
                    actions, action_idx = torch.topk(pred, 75)
                    actions = actions.tolist()[0]
                    action_idx = action_idx.tolist()[0]
                    pred_action = actions_i_to_x[action_idx[0]]
                    # print("Pred action", pred_action)

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
        
        uas, las = compute_metrics(words_list, gold, all_pred, 2)

        print(f"Test UAS at epoch {epoch + 1}: ", uas)
        print(f"Test LAS at epoch {epoch + 1}: ", las) 
        


if __name__ == "__main__":
    __main__()
    