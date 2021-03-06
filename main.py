import argparse
from typing import overload

import ltn
import numpy as np
import torch
from datasets import Dataset
from datasets import load_dataset, load_metric
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.activations import get_activation, ACT2FN

from kobert_tokenizer import KoBERTTokenizer

# global constants
gpu_ids = tuple([n for n in range(torch.cuda.device_count())])
cuda_ids = tuple([f"cuda:{n}" for n in gpu_ids])
data_files = {
    "cls": {
        "train": "data/klue-sts-cls/train.json",
        "valid": "data/klue-sts-cls/valid.json",
    },
    "reg": {
        "train": "data/klue-sts-reg/train.json",
        "valid": "data/klue-sts-reg/valid.json",
    },
}
lang_models = (
    "bert-base-multilingual-uncased",
    "skt/kobert-base-v1",
    "monologg/koelectra-base-v3-discriminator",
    "monologg/kobigbird-bert-base",
)


class DataLoader(object):
    """PyTorch DataLoader to load the dataset for the training and testing of the model"""

    def __init__(self, input_ids, attention_mask, token_type_ids, labels, batch_size=1, shuffle=False):
        self.input_ids = torch.tensor(input_ids)
        self.attention_mask = torch.tensor(attention_mask)
        self.token_type_ids = torch.tensor(token_type_ids)
        self.labels = torch.tensor(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.input_ids.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.input_ids.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            yield (
                self.input_ids[idxlist[start_idx:end_idx]],
                self.attention_mask[idxlist[start_idx:end_idx]],
                self.token_type_ids[idxlist[start_idx:end_idx]],
                self.labels[idxlist[start_idx:end_idx]]
            )


class BertClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks.
    - Almost same as `transformers.models.electra.modeling_electra.ElectraClassificationHead`
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("tanh")(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks.
    - Same as `transformers.models.electra.modeling_electra.ElectraClassificationHead`
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BigBirdClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks.
    - Same as `transformers.models.big_bird.modeling_big_bird.BigBirdClassificationHead`
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TrainerBase(object):
    """LTN-based Trainer (abstract class)"""

    def __init__(self, lang_model, task_type, gpu_id=0,
                 max_epoch=1, num_train_sample=-1, num_test_sample=50,
                 learning_rate=2e-5, batch_size=8, max_seq_length=512,
                 num_check_tokenized=1, check_tokenizer=True, check_pretrained=True):
        # save parameters
        self.lang_model = lang_model
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_check_tokenized = num_check_tokenized
        self.check_tokenizer = check_tokenizer
        self.check_pretrained = check_pretrained

        # set cuda device
        self.device = torch.device(cuda_ids[gpu_id] if gpu_id in gpu_ids and torch.cuda.is_available() else "cpu")
        ltn.device = self.device
        print("\n" + "=" * 112)
        print(f"[device] {self.device} ??? [{', '.join(cuda_ids)}]")
        print("=" * 112 + "\n")

        # load raw datasets
        raw_datasets = load_dataset("json", data_files=data_files[task_type], field="data")
        if num_train_sample is not None and num_test_sample is not None and num_train_sample > 0 and num_test_sample > 0:
            for k in raw_datasets.keys():
                raw_datasets[k] = Dataset.from_dict(raw_datasets[k][:(num_train_sample if k == "train" else num_test_sample)])
        print("\n" + "=" * 112)
        print(f'[raw_datasets] {raw_datasets}')
        self.text1_key, self.text2_key = "sentence1", "sentence2"
        print(f'- input_columns: {self.text1_key}, {self.text2_key}')
        print("=" * 112 + "\n")

        # load tokenizer
        self.tokenizer = KoBERTTokenizer.from_pretrained(lang_model) if "kobert" in lang_model \
            else AutoTokenizer.from_pretrained(lang_model)
        print("\n" + "=" * 112)
        print(f'[tokenizer({type(self.tokenizer).__name__})] {self.tokenizer}')
        if self.check_tokenizer:
            text = "[CLS] ????????? ???????????? ????????? ???????????????. [SEP]"
            tokens = self.tokenizer.tokenize(text)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            print("- text   =", text)
            print("- tokens =", tokens)
            print("- ids    =", ids)
        print("=" * 112 + "\n")

        # tokenize texts in datasets and make loaders
        self.token_printing_counter = 0
        raw_datasets = raw_datasets.map(self.batch_tokenize, batched=True, batch_size=2000, num_proc=1,
                                        load_from_cache_file=False, desc="Running tokenizer on dataset")
        train_dataset = raw_datasets["train"]
        valid_dataset = raw_datasets["valid"]
        test_dataset = raw_datasets["test"] if "test" in raw_datasets else None
        self.train_loader = DataLoader(train_dataset['input_ids'], train_dataset['attention_mask'],
                                       train_dataset['token_type_ids'], train_dataset['label'],
                                       batch_size=batch_size, shuffle=False)
        self.valid_loader = DataLoader(valid_dataset['input_ids'], valid_dataset['attention_mask'],
                                       valid_dataset['token_type_ids'], valid_dataset['label'],
                                       batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset['input_ids'], test_dataset['attention_mask'],
                                      test_dataset['token_type_ids'], test_dataset['label'],
                                      batch_size=batch_size, shuffle=False) if test_dataset is not None else None

    def batch_tokenize(self, examples):
        args = (
            (examples[self.text1_key],) if self.text2_key is None else (examples[self.text1_key], examples[self.text2_key])
        )
        result = self.tokenizer(*args, padding='max_length', max_length=self.max_seq_length, truncation=True)
        if self.token_printing_counter < self.num_check_tokenized:
            print("\n" + "=" * 112)
            for i, a in enumerate(result['input_ids'][:1]):
                tokens = self.tokenizer.convert_ids_to_tokens(a)
                print(f"- [tokens]({len(a)})\t= {' '.join(tokens[:25])} ... {' '.join(tokens[-25:])}")
                self.token_printing_counter += 1
            print("=" * 112 + "\n")
        return result

    def sample_pretrained(self, model):
        model_desc = str(model).splitlines()
        idx1 = next((i for i, x in enumerate(model_desc) if "(encoder)" in x), 8)
        idx2 = next((i for i, x in enumerate(model_desc) if "(pooler)" in x), -1)
        print("\n" + "=" * 112)
        print(f'[pretrained] {chr(10).join(model_desc[:idx1] + ["  ..."] + model_desc[idx2:])}')
        if self.check_pretrained:
            batch_text = ["????????? ???????????? ????????? ???????????????.", "????????? ????????? ??????."]
            inputs = self.tokenizer.batch_encode_plus(batch_text, padding='max_length', max_length=self.max_seq_length, truncation=True)
            output = model(
                torch.tensor(inputs['input_ids']),
                torch.tensor(inputs['attention_mask']),
                torch.tensor(inputs['token_type_ids'])
            )
            hidden = output.last_hidden_state
            print(f"-      input_ids({'x'.join(str(x) for x in list(torch.tensor(inputs['input_ids']).size()))}) : [{', '.join(str(x) for x in inputs['input_ids'][0][:25])}, ..., {', '.join(str(x) for x in inputs['input_ids'][0][-25:])}]")
            print(f"- attention_mask({'x'.join(str(x) for x in list(torch.tensor(inputs['attention_mask']).size()))}) : [{', '.join(str(x) for x in inputs['attention_mask'][0][:25])}, ..., {', '.join(str(x) for x in inputs['attention_mask'][0][-25:])}]")
            print(f"- token_type_ids({'x'.join(str(x) for x in list(torch.tensor(inputs['token_type_ids']).size()))}) : [{', '.join(str(x) for x in inputs['token_type_ids'][0][:25])}, ..., {', '.join(str(x) for x in inputs['token_type_ids'][0][-25:])}]")
            print(f"-  output_hidden({'x'.join(str(x) for x in list(hidden.size()))}) : {hidden[0]}")
        print("=" * 112 + "\n")

    @overload
    def define(self):
        ...

    @overload
    def train(self):
        ...

    @overload
    def measure(self, loader, f):
        ...


class TrainerForClassification(TrainerBase):
    """LTN-based Trainer for classification"""

    def __init__(self, **kwargs):
        super().__init__(task_type="cls", **kwargs)

    class Net(nn.Module):
        def __init__(self, trainer):
            super().__init__()
            config = AutoConfig.from_pretrained(trainer.lang_model, num_labels=1)
            self.middle = AutoModel.from_pretrained(trainer.lang_model, config=config)
            self.topper = BigBirdClassificationHead(config) if config.model_type == "big_bird" \
                else ElectraClassificationHead(config) if config.model_type == "electra" \
                else BertClassificationHead(config)
            self.sigmoid = nn.Sigmoid()
            trainer.sample_pretrained(self.middle)

        def forward(self, x1, x2, x3):
            output = self.middle(
                torch.squeeze(x1, dim=1) if x1.size(dim=1) == 1 and len(x1.size()) == 3 else x1,
                torch.squeeze(x2, dim=1) if x2.size(dim=1) == 1 and len(x2.size()) == 3 else x2,
                torch.squeeze(x3, dim=1) if x3.size(dim=1) == 1 and len(x3.size()) == 3 else x3,
            )
            hidden = output.last_hidden_state
            logits = self.topper(hidden)
            probs = self.sigmoid(logits)
            return probs

    def define(self):
        # LTN setting #1
        predicate = ltn.Predicate(model=TrainerForClassification.Net(trainer=self)).to(self.device)
        self.ltn = predicate

        # optimizer setting
        self.optimizer = torch.optim.Adam(self.ltn.parameters(), lr=self.learning_rate)

        # metric setting
        self.metric = load_metric("glue", "mrpc")
        print("\n" + "=" * 112)
        print(f'[metric] {", ".join(self.metric.compute(predictions=[0, 1], references=[0, 1]).keys())}')
        print("=" * 112 + "\n")
        return self

    def train(self):
        # LTN setting #2
        Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        ForAll = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        SatAgg = ltn.fuzzy_ops.SatAgg()

        # training of the predicate using a loss containing the satisfaction level of the knowledge base
        # the objective is to maximize the satisfaction level of the knowledge base
        for epoch in range(self.max_epoch):
            train_loss = 0.0
            for batch in self.train_loader:
                x1, x2, x3 = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                y = batch[3] >= 0.5
                self.optimizer.zero_grad()

                # ground the variables with current batch data
                p1 = ltn.Variable("p1", x1[torch.nonzero(y)])  # positive examples
                p2 = ltn.Variable("p2", x2[torch.nonzero(y)])  # positive examples
                p3 = ltn.Variable("p3", x3[torch.nonzero(y)])  # positive examples
                q1 = ltn.Variable("q1", x1[torch.nonzero(torch.logical_not(y))])  # negative examples
                q2 = ltn.Variable("q2", x2[torch.nonzero(torch.logical_not(y))])  # negative examples
                q3 = ltn.Variable("q3", x3[torch.nonzero(torch.logical_not(y))])  # negative examples

                # calculate loss and backpropagate
                loss = 1.0 - SatAgg(
                    ForAll(ltn.diag(p1, p2, p3), self.ltn(p1, p2, p3)),
                    ForAll(ltn.diag(q1, q2, q3), Not(self.ltn(q1, q2, q3)))
                )
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # trace metrics at every epoch
            train_loss = train_loss / len(self.train_loader)
            train_score = self.measure(self.train_loader, f=self.ltn.model)
            valid_score = self.measure(self.valid_loader, f=self.ltn.model)
            test_score = self.measure(self.test_loader, f=self.ltn.model) if self.test_loader is not None else None
            print(' | '.join(x for x in [
                f"Epoch {epoch + 1:02d}", f"Loss {train_loss:.6f}",
                f"Train {', '.join(f'{k}={v:.4f}' for k, v in train_score.items())}",
                f"Valid {', '.join(f'{k}={v:.4f}' for k, v in valid_score.items())}",
                f"Test {', '.join(f'{k}={v:.4f}' for k, v in test_score.items())}" if test_score is not None else None,
            ] if x is not None))

        return 0

    def measure(self, loader, f):
        ps = []
        ys = []
        for batch in loader:
            x1, x2, x3 = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            y = batch[3].detach().numpy()
            p = f(x1, x2, x3).cpu().detach().numpy()
            p = np.where(p >= 0.5, 1, 0)
            p = np.squeeze(p)
            ps.extend(p)
            ys.extend(y)
        return self.metric.compute(predictions=ps, references=ys)


class TrainerForRegression(TrainerBase):
    """LTN-based Trainer for regression"""

    def __init__(self, **kwargs):
        super().__init__(task_type="reg", **kwargs)

    class Net(nn.Module):
        def __init__(self, trainer):
            super().__init__()
            config = AutoConfig.from_pretrained(trainer.lang_model, num_labels=1)
            self.middle = AutoModel.from_pretrained(trainer.lang_model, config=config)
            self.topper = BigBirdClassificationHead(config) if config.model_type == "big_bird" \
                else ElectraClassificationHead(config) if config.model_type == "electra" \
                else BertClassificationHead(config)
            trainer.sample_pretrained(self.middle)

        def forward(self, x1, x2, x3):
            output = self.middle(
                torch.squeeze(x1, dim=1) if x1.size(dim=1) == 1 and len(x1.size()) == 3 else x1,
                torch.squeeze(x2, dim=1) if x2.size(dim=1) == 1 and len(x2.size()) == 3 else x2,
                torch.squeeze(x3, dim=1) if x3.size(dim=1) == 1 and len(x3.size()) == 3 else x3,
            )
            hidden = output.last_hidden_state
            logits = self.topper(hidden)
            return logits

    def define(self):
        # LTN setting #1
        function = ltn.Function(model=TrainerForRegression.Net(trainer=self)).to(self.device)
        self.ltn = function

        # optimizer setting
        self.optimizer = torch.optim.Adam(self.ltn.parameters(), lr=self.learning_rate)

        # metric setting
        self.metric = load_metric("glue", "stsb")
        print("\n" + "=" * 112)
        print(f'[metric] {", ".join(self.metric.compute(predictions=[0, 1], references=[0, 1]).keys())}')
        print("=" * 112 + "\n")
        return self

    def train(self):
        # LTN setting #2
        alpha = 0.05
        Eq = ltn.Predicate(func=lambda u, v: torch.exp(-alpha * torch.sqrt(torch.sum(torch.square(u - v), dim=1))))
        ForAll = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        SatAgg = ltn.fuzzy_ops.SatAgg()

        # training of the predicate using a loss containing the satisfaction level of the knowledge base
        # the objective is to maximize the satisfaction level of the knowledge base
        for epoch in range(self.max_epoch):
            train_loss = 0.0
            for batch in self.train_loader:
                x1, x2, x3 = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                y = batch[3]
                self.optimizer.zero_grad()

                # ground the variables with current batch data
                x1 = ltn.Variable("x1", x1)  # samples
                x2 = ltn.Variable("x2", x2)  # samples
                x3 = ltn.Variable("x3", x3)  # samples
                y = ltn.Variable("y", y)  # ground truths

                # calculate loss and backpropagate
                loss = 1.0 - SatAgg(
                    ForAll(ltn.diag(x1, x2, x3, y), Eq(self.ltn(x1, x2, x3), y))
                )
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # trace metrics at every epoch
            train_loss = train_loss / len(self.train_loader)
            train_score = self.measure(self.train_loader, f=self.ltn.model)
            valid_score = self.measure(self.valid_loader, f=self.ltn.model)
            test_score = self.measure(self.test_loader, f=self.ltn.model) if self.test_loader is not None else None
            print(' | '.join(x for x in [
                f"Epoch {epoch + 1:02d}", f"Loss {train_loss:.6f}",
                f"Train {', '.join(f'{k}={v:.4f}' for k, v in train_score.items())}",
                f"Valid {', '.join(f'{k}={v:.4f}' for k, v in valid_score.items())}",
                f"Test {', '.join(f'{k}={v:.4f}' for k, v in test_score.items())}" if test_score is not None else None,
            ] if x is not None))

        return 0

    def measure(self, loader, f):
        ps = []
        ys = []
        for batch in loader:
            x1, x2, x3 = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            y = batch[3].detach().numpy()
            p = f(x1, x2, x3).cpu().detach().numpy()
            p = np.squeeze(p)
            ps.extend(p)
            ys.extend(y)
        return self.metric.compute(predictions=ps, references=ys)


# main entry
if __name__ == '__main__':
    expr_tasks = {
        "STS-CLS": lambda n, m, k, e, lr, bs, msl: sum([
            TrainerForClassification(gpu_id=n, lang_model=lang_models[m], num_train_sample=k, max_epoch=e,
                                     learning_rate=lr, batch_size=bs, max_seq_length=msl).define().train(),
        ]),
        "STS-REG": lambda n, m, k, e, lr, bs, msl: sum([
            TrainerForRegression(gpu_id=n, lang_model=lang_models[m], num_train_sample=k, max_epoch=e,
                                 learning_rate=lr, batch_size=bs, max_seq_length=msl).define().train(),
        ]),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", default=None, type=str, required=True,
                        help=f"task name: {', '.join(expr_tasks.keys())}")
    parser.add_argument("-n", default=0, type=int, required=False,
                        help=f"gpu id: {', '.join(str(x) for x in gpu_ids)}")
    parser.add_argument("-m", default=0, type=int, required=False,
                        help=f"pretrained model id: {', '.join(str(x) for x in range(len(lang_models)))}")
    parser.add_argument("-k", default=100, type=int, required=False, help=f"number of training samples")
    parser.add_argument("-e", default=10, type=int, required=False, help=f"number of training epochs")
    parser.add_argument("-lr", default=2e-5, type=float, required=False, help=f"learning rate")
    parser.add_argument("-bs", default=8, type=int, required=False, help=f"batch size")
    parser.add_argument("-msl", default=512, type=int, required=False, help=f"max sequence length")
    args = parser.parse_args()

    if args.t in expr_tasks:
        r = expr_tasks[args.t](n=args.n, m=args.m, k=args.k, e=args.e, lr=args.lr, bs=args.bs, msl=args.msl)
        if r > 0:
            raise ValueError(f"Error code returned: {r}")
    else:
        raise ValueError(f"Not available task: {args.t}")
