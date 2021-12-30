import ltn
import numpy as np
import torch
from datasets import Dataset
from datasets import load_dataset, load_metric
from transformers import ElectraConfig, ElectraModel, ElectraTokenizer
from transformers.models.electra.modeling_electra import ElectraClassificationHead


# PyTorch DataLoader to load the dataset for the training and testing of the model
class DataLoader(object):
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


token_printing_counter = 0


# train LTN model and test
def do_experiment(
        max_epoch, pretrained, data_files, n_gpu=0,
        num_train_sample=None, num_test_sample=50,
        max_seq_length=512, learning_rate=1e-5, batch_size=8,
        num_check_tokenized=1, check_tokenizer=False, check_pretrained=False,
):
    # set cuda device
    gpu_ids = [f"cuda:{n}" for n in range(torch.cuda.device_count())]
    device = torch.device(gpu_ids[n_gpu % len(gpu_ids)] if torch.cuda.is_available() else "cpu")
    ltn.device = device
    print("\n" + "=" * 112)
    print(f"[device] {device} ∈ [{', '.join(gpu_ids)}]")
    print("=" * 112 + "\n")

    # load raw datasets
    raw_datasets = load_dataset("json", data_files=data_files, field="data")
    if num_train_sample is not None and num_test_sample is not None:
        for k in raw_datasets.keys():
            raw_datasets[k] = Dataset.from_dict(raw_datasets[k][:(num_train_sample if k == "train" else num_test_sample)])
    print("\n" + "=" * 112)
    print(f'[raw_datasets] {raw_datasets}')
    text1_key, text2_key = "sentence1", "sentence2"
    print(f'- input_columns: {text1_key}, {text2_key}')
    print("=" * 112 + "\n")

    # load tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(pretrained)
    print("\n" + "=" * 112)
    print(f'[tokenizer] {tokenizer}')
    if check_tokenizer:
        text = "[CLS] 한국어 ELECTRA를 공유합니다. [SEP]"
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print("- text   =", text)
        print("- tokens =", tokens)
        print("- ids    =", ids)
    print("=" * 112 + "\n")

    # tokenize texts in the given examples
    def batch_tokenize(examples):
        global token_printing_counter
        args = (
            (examples[text1_key],) if text2_key is None else (examples[text1_key], examples[text2_key])
        )
        result = tokenizer(*args, padding='max_length', max_length=max_seq_length, truncation=True)
        if token_printing_counter < num_check_tokenized:
            print("\n" + "=" * 112)
            for i, a in enumerate(result['input_ids'][:1]):
                print(f"- [tokens]({len(a)})\t= {tokenizer.convert_ids_to_tokens(a)}")
                token_printing_counter += 1
            print("=" * 112 + "\n")
        return result

    # tokenize texts in datasets and make loaders
    global token_printing_counter
    token_printing_counter = 0
    raw_datasets = raw_datasets.map(batch_tokenize, batched=True, batch_size=2000, num_proc=1,
                                    load_from_cache_file=False, desc="Running tokenizer on dataset")
    train_dataset = raw_datasets["train"]
    valid_dataset = raw_datasets["valid"]
    test_dataset = raw_datasets["test"] if "test" in raw_datasets else None
    train_loader = DataLoader(train_dataset['input_ids'], train_dataset['attention_mask'],
                              train_dataset['token_type_ids'], train_dataset['label'],
                              batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset['input_ids'], valid_dataset['attention_mask'],
                              valid_dataset['token_type_ids'], valid_dataset['label'],
                              batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset['input_ids'], test_dataset['attention_mask'],
                             test_dataset['token_type_ids'], test_dataset['label'],
                             batch_size=batch_size, shuffle=False) if test_dataset is not None else None

    # Neural network using ELECTRA for binary classification task
    class ElectraClassificationNet(torch.nn.Module):
        def __init__(self, num_labels=1):
            super(ElectraClassificationNet, self).__init__()
            self.sigmoid = torch.nn.Sigmoid()
            config = ElectraConfig.from_pretrained(pretrained, num_labels=num_labels)
            self.classifier = ElectraClassificationHead(config)
            self.electra = ElectraModel.from_pretrained(pretrained, config=config)
            print("\n" + "=" * 112)
            print(f'[pretrained] {chr(10).join(str(self.electra).splitlines()[:8])}')
            if check_pretrained:
                batch_text = ["한국어 모델을 공유합니다.", "오늘은 날씨가 좋다."]
                inputs = tokenizer.batch_encode_plus(batch_text, padding='max_length', max_length=max_seq_length, truncation=True)
                hidden = self.electra(
                    torch.tensor(inputs['input_ids']),
                    torch.tensor(inputs['attention_mask']),
                    torch.tensor(inputs['token_type_ids'])
                ).last_hidden_state
                print(f"-      input_ids({list(torch.tensor(inputs['input_ids']).size())}) : {inputs['input_ids'][0]}")
                print(f"- attention_mask({list(torch.tensor(inputs['attention_mask']).size())}) : {inputs['attention_mask'][0]}")
                print(f"- token_type_ids({list(torch.tensor(inputs['token_type_ids']).size())}) : {inputs['token_type_ids'][0]}")
                print(f"-   hidden_state({list(hidden.size())}) : {hidden[0]}")
            print("=" * 112 + "\n")

        def forward(self, x1, x2, x3):
            hidden = self.electra(
                torch.squeeze(x1, dim=1) if x1.size(dim=1) == 1 and len(x1.size()) == 3 else x1,
                torch.squeeze(x2, dim=1) if x2.size(dim=1) == 1 and len(x2.size()) == 3 else x2,
                torch.squeeze(x3, dim=1) if x3.size(dim=1) == 1 and len(x3.size()) == 3 else x3,
            ).last_hidden_state
            logits = self.classifier(hidden)
            probs = self.sigmoid(logits)
            return probs

    # LTN setting
    predicate = ltn.Predicate(ElectraClassificationNet(num_labels=1)).to(device)
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    ForAll = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    # optimizer setting
    optimizer = torch.optim.Adam(predicate.parameters(), lr=learning_rate)

    # metric setting
    metric = load_metric("glue", "mrpc")
    print("\n" + "=" * 112)
    print(f'[metric] {metric.info.metric_name}/{metric.info.config_name}')
    print(f'- {metric.info.description.strip().splitlines()[0]}')
    print("=" * 112 + "\n")

    # computes the overall metric of the predictions
    def compute_metric(loader):
        ps = []
        ys = []
        for batch in loader:
            f = predicate.model
            x1, x2, x3 = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            y = batch[3].detach().numpy()
            p = f(x1, x2, x3).cpu().detach().numpy()
            p = np.where(p >= 0.5, 1, 0).flatten()
            ps.extend(p)
            ys.extend(y)
        return metric.compute(predictions=ps, references=ys)

    # training of the predicate using a loss containing the satisfaction level of the knowledge base
    # the objective is to maximize the satisfaction level of the knowledge base
    for epoch in range(max_epoch):
        train_loss = 0.0
        for batch in train_loader:
            f = predicate
            x1, x2, x3 = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            y = batch[3] >= 0.5
            optimizer.zero_grad()

            # ground the variables with current batch data
            p1 = ltn.Variable("p1", x1[torch.nonzero(y)])  # positive examples
            p2 = ltn.Variable("p2", x2[torch.nonzero(y)])  # positive examples
            p3 = ltn.Variable("p3", x3[torch.nonzero(y)])  # positive examples
            q1 = ltn.Variable("q1", x1[torch.nonzero(torch.logical_not(y))])  # negative examples
            q2 = ltn.Variable("q2", x2[torch.nonzero(torch.logical_not(y))])  # negative examples
            q3 = ltn.Variable("q3", x3[torch.nonzero(torch.logical_not(y))])  # negative examples

            # calculate loss and backpropagate
            loss = 1.0 - SatAgg(
                ForAll(ltn.diag(p1, p2, p3), f(p1, p2, p3)),
                ForAll(ltn.diag(q1, q2, q3), Not(f(q1, q2, q3)))
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # trace metrics at every epoch
        train_loss = train_loss / len(train_loader)
        train_score = compute_metric(train_loader)
        valid_score = compute_metric(valid_loader)
        test_score = compute_metric(test_loader) if test_loader is not None else None
        print(' | '.join(x for x in [
            f"Epoch {epoch + 1:02d}", f"Loss {train_loss:.6f}",
            f"Train {', '.join(f'{k[:3]}={v:.4f}' for k, v in train_score.items())}",
            f"Valid {', '.join(f'{k[:3]}={v:.4f}' for k, v in valid_score.items())}",
            f"Test {', '.join(f'{k[:3]}={v:.4f}' for k, v in test_score.items())}" if test_score is not None else None,
        ] if x is not None))


# main entry
if __name__ == '__main__':
    data_files = {
        "train": "data/klue-sts-cls/train.json",
        "valid": "data/klue-sts-cls/valid.json",
    }
    do_experiment(data_files=data_files,
                  pretrained="monologg/kobigbird-bert-base",
                  max_epoch=10, max_seq_length=512, num_train_sample=100)
