'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentenceConcatenatedPairDataset,
    SentenceConcatenatedPairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.polar_sentiment_layer = nn.Linear(config.hidden_size, 1)
        self.sentiment_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.paraphrase_layer = nn.Linear(config.hidden_size, 1)
        self.similarity_layer = nn.Linear(config.hidden_size, 1)

        self.bert_cache = {}
        self.pretrain = config.option == 'pretrain'

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        return self.bert(input_ids, attention_mask)['pooler_output']

    def predict_sentiment(self, input_ids, attention_mask, sent_ids):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        bert_embedding = None
        if self.pretrain:
            bert_encodings = []
            for i in range(len(sent_ids)):
                if sent_ids[i] not in self.bert_cache:
                    self.bert_cache[sent_ids[i]] = self.forward(input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0))
                bert_encodings.append(self.bert_cache[sent_ids[i]].flatten())
            bert_embedding = torch.stack(bert_encodings)
        else:
            bert_embedding = self.forward(input_ids, attention_mask)
        logits = F.softmax(self.sentiment_layer(bert_embedding), dim=1)
        return logits

    def predict_paraphrase(self, input_ids, attention_mask, sent_ids):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        sent_embedding = None
        if self.pretrain:
            bert_encodings = []
            for i in range(len(sent_ids)):
                if sent_ids[i] not in self.bert_cache:
                    self.bert_cache[sent_ids[i]] = self.forward(input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0))
                bert_encodings.append(self.bert_cache[sent_ids[i]].flatten())
            sent_embedding = torch.stack(bert_encodings)
        else:
            sent_embedding = self.forward(input_ids, attention_mask)
        logits = self.paraphrase_layer(sent_embedding)
        return logits

    def predict_similarity(self, input_ids, attention_mask, sent_ids):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        sent_embedding = None
        if self.pretrain:
            bert_encodings = []
            for i in range(len(sent_ids)):
                if sent_ids[i] not in self.bert_cache:
                    self.bert_cache[sent_ids[i]] = self.forward(input_ids[i].unsqueeze(0),
                                                                attention_mask[i].unsqueeze(0))
                bert_encodings.append(self.bert_cache[sent_ids[i]].flatten())
            sent_embedding = torch.stack(bert_encodings)
        else:
            sent_embedding = self.forward(input_ids, attention_mask)
        logits = self.similarity_layer(sent_embedding)
        return logits

    def predict_polar_sentiment(self, input_ids, attention_mask, sent_ids):
        sent_embedding = None
        if self.pretrain:
            bert_encodings = []
            for i in range(len(sent_ids)):
                if sent_ids[i] not in self.bert_cache:
                    self.bert_cache[sent_ids[i]] = self.forward(input_ids[i].unsqueeze(0),
                                                                attention_mask[i].unsqueeze(0))
                bert_encodings.append(self.bert_cache[sent_ids[i]].flatten())
            sent_embedding = torch.stack(bert_encodings)
        else:
            sent_embedding = self.forward(input_ids, attention_mask)
        logits = self.polar_sentiment_layer(sent_embedding)
        return logits


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels, para_train_data, sts_train_data, cfimdb_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, args.cfimdb_train, split ='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data, cfimdb_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, args.cfimdb_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentenceConcatenatedPairDataset(para_train_data, args)
    para_dev_data = SentenceConcatenatedPairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentenceConcatenatedPairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentenceConcatenatedPairDataset(sts_dev_data, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=sts_dev_data.collate_fn)

    cfimdb_train_data = SentenceClassificationDataset(cfimdb_train_data, args)
    cfimdb_dev_data = SentenceClassificationDataset(cfimdb_dev_data, args)

    cfimdb_train_dataloader = DataLoader(cfimdb_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=cfimdb_train_data.collate_fn)
    cfimdb_dev_dataloader = DataLoader(cfimdb_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=cfimdb_dev_data.collate_fn)
    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': len(num_labels),
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    # Run for the specified number of epochs.
    para_data_subsets = random_split(para_train_data, [1/17] * 17)
    best_dev_score = 0
    for epoch in range(args.epochs):
        model.train()

        para_train_dataloader = DataLoader(para_data_subsets[epoch % len(para_data_subsets)],
                                           shuffle=True, batch_size=args.batch_size,
                                           collate_fn=para_train_data.collate_fn)
        polar_loss = 0
        if args.use_cfimdb:
            polar_loss = train_polar_sentiment(model, epoch, args.batch_size, optimizer, device, cfimdb_train_dataloader)
        sst_loss = train_sentiment(model, epoch, args.batch_size, optimizer, device, sst_train_dataloader)
        para_loss = train_paraphrase(model, epoch, args.batch_size, optimizer, device, para_train_dataloader)
        sts_loss = train_similarity(model, epoch, args.batch_size, optimizer, device, sts_train_dataloader)

        sst_train_acc, _, _, para_train_acc, _, _, sts_train_corr, *_ = model_eval_multitask(sst_train_dataloader,
                                                                                             para_train_dataloader,
                                                                                             sts_train_dataloader,
                                                                                             model, device)
        sst_dev_acc, _, _, para_dev_acc, _, _, sts_dev_corr, *_ = model_eval_multitask(sst_dev_dataloader,
                                                                                       para_dev_dataloader,
                                                                                       sts_dev_dataloader,
                                                                                       model, device)
        train_mean_score = (sst_train_acc + para_train_acc + sts_train_corr) / 3
        dev_mean_score = (sst_dev_acc + para_dev_acc + sts_dev_corr) / 3
        if dev_mean_score > best_dev_score:
            best_dev_score = dev_mean_score
            save_model(model, optimizer, args, config, args.filepath)

        print(
            f"Epoch {epoch}: sentiment loss :: {sst_loss :.3f},\n"
            f" paraphrase loss :: {para_loss :.3f},\n"
            f" similarity loss :: {sts_loss :.3f},\n"
            f" polar similarity loss :: {polar_loss :.3f},\n"
            f" sentiment train acc :: {sst_train_acc :.3f},\n"
            f" paraphrase train acc :: {para_train_acc :.3f},\n"
            f" similarity train corr :: {sts_train_corr :.3f},\n"
            f" sentiment dev acc :: {sst_dev_acc :.3f},\n"
            f" paraphrase dev acc :: {para_dev_acc :.3f},\n"
            f" similarity dev corr :: {sts_dev_corr :.3f},\n"
            f" train mean score :: {train_mean_score :.3f}, "
            f"dev mean score :: {dev_mean_score :.3f}")


def train_sentiment(model, epoch, batch_size, optimizer, device, sst_train_dataloader):
    print('Training on SST...')
    train_loss = 0
    num_batches = 0
    for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask, b_labels, b_sent_ids = (batch['token_ids'],
                                    batch['attention_mask'], batch['labels'], batch['sent_ids'])

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)

        optimizer.zero_grad()
        logits = model.predict_sentiment(b_ids, b_mask, b_sent_ids)
        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / batch_size

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    return train_loss / (num_batches)


def train_paraphrase(model, epoch, batch_size, optimizer, device, para_train_dataloader):
    print('Training on Quora...')
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask, b_labels, b_sent_ids = (batch['token_ids'],
                                               batch['attention_mask'], batch['labels'], batch['sent_ids'])

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)

        optimizer.zero_grad()
        logits = model.predict_paraphrase(b_ids, b_mask, b_sent_ids).sigmoid().flatten()
        loss = F.binary_cross_entropy(logits, b_labels.view(-1).float(), reduction='sum') / batch_size

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    return train_loss / num_batches


def train_similarity(model, epoch, batch_size, optimizer, device, sts_train_dataloader):
    print('Training on STS...')
    train_loss = 0
    num_batches = 0
    for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask, b_labels, b_sent_ids = (batch['token_ids'],
                                               batch['attention_mask'], batch['labels'], batch['sent_ids'])

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)

        optimizer.zero_grad()
        logits = model.predict_similarity(b_ids, b_mask, b_sent_ids).flatten()
        loss = F.mse_loss(logits, b_labels.view(-1).float(), reduction='sum') / batch_size

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    return train_loss / num_batches


def train_polar_sentiment(model, epoch, batch_size, optimizer, device, cfimdb_train_dataloader):
    print('Training on CFIMDB...')
    train_loss = 0
    num_batches = 0
    for batch in tqdm(cfimdb_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask, b_labels, b_sent_ids = (batch['token_ids'],
                                               batch['attention_mask'], batch['labels'], batch['sent_ids'])

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)

        optimizer.zero_grad()
        logits = model.predict_polar_sentiment(b_ids, b_mask, b_sent_ids).sigmoid().flatten()
        loss = F.binary_cross_entropy(logits, b_labels.view(-1).float(), reduction='sum') / batch_size

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    return train_loss / num_batches


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath, map_location=device)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data, _ = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, args.cfimdb_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data, _ = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, args.cfimdb_dev, split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentenceConcatenatedPairTestDataset(para_test_data, args)
        para_dev_data = SentenceConcatenatedPairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentenceConcatenatedPairTestDataset(sts_test_data, args)
        sts_dev_data = SentenceConcatenatedPairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+", encoding='utf8') as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+", encoding='utf8') as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+", encoding='utf8') as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+", encoding='utf8') as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+", encoding='utf8') as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+", encoding='utf8') as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--cfimdb_train", type=str, default="data/ids-cfimdb-train.csv")
    parser.add_argument("--cfimdb_dev", type=str, default="data/ids-cfimdb-dev.csv")
    parser.add_argument("--cfimdb_test", type=str, default="data/ids-cfimdb-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated; evaluate: the model is evaluated on dev and test datasets',
                        choices=('pretrain', 'finetune', 'evaluate'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--use_cfimdb", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--filepath", type=str, help="model path", default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if not args.filepath:
        args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    if args.option == 'evaluate':
        test_multitask(args)
    else:
        train_multitask(args)

