import os
import random

from torch.utils.data import DataLoader,ConcatDataset
import lightning.pytorch as pl
from datasets import load_dataset, load_from_disk


from transformers import AutoTokenizer, default_data_collator


class OICDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        dataset: str,
        preprocessed_dir: str = None,
        model_name_or_path: str = None,
        known_cls_ratio: float = 0.5,
        labeled_ratio: float = 1.0,
        include_unknown: bool = False,
        max_seq_len: int = 45,
        batch_size: int = 32,
        num_workers: int = 8,
        k_1:bool=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_path = os.path.join(data_path, dataset)
        self.train_data_path = f"{self.data_path}/train.tsv"
        self.val_data_path = f"{self.data_path}/dev.tsv"
        self.test_data_path = f"{self.data_path}/test.tsv"

        self.model_name_or_path = model_name_or_path

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.known_cls_ratio = known_cls_ratio
        self.labeled_ratio = labeled_ratio
        self.include_unknown = include_unknown
        self.num_classes = None  # set after preparing
        self.pre_dir = preprocessed_dir
        self.k_1 = k_1
        print("self.k_1", k_1)
        if k_1 != False:
            self.k_1 = k_1

    def prepare_data(self):
        if self.pre_dir and os.path.isdir(self.pre_dir):
            return

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True
        )
        if self.k_1 !=False:
            print("neggggggg")
            data_files = {
                "train": self.train_data_path,
                "validation": self.val_data_path,
                "test": self.test_data_path,
                "neg": 'data/SQUAD/squad.tsv', 
            }
        else:
            print("basicc")
            data_files = {
                "train": self.train_data_path,
                "validation": self.val_data_path,
                "test": self.test_data_path,
            }
        
        raw_datasets = load_dataset("csv", data_files=data_files, delimiter="\t")

        all_label_list = raw_datasets["train"].unique("label")
        n_known_cls = round(len(all_label_list) * self.known_cls_ratio)
        known_label_list = random.sample(all_label_list, k=n_known_cls)
        known_label_list.sort()  # Sort it for determinism
        self.num_classes = len(known_label_list)
        self.unseen_label_id = self.num_classes

        label_to_id = {v: i for i, v in enumerate(known_label_list)}
        self.label_to_id = label_to_id
        self.pre_dir = f"data/preprocessed/{self.dataset}_{self.known_cls_ratio}"

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["text"],
            fn_kwargs={"label_to_id": self.label_to_id, "tokenizer": tokenizer},
            load_from_cache_file=False,  # not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        # seed value will be added
        raw_datasets.save_to_disk(self.pre_dir)
        # breakpoint()
    def setup(self, stage):
        ds = load_from_disk(self.pre_dir)
        if stage in (None, "fit"):
            if self.k_1 != False:
                print("neg")
                self.neg = ds['neg']
            ds = ds.filter(lambda x: x["label"] != self.unseen_label_id)
            self.train = ds["train"]
            self.validation = ds["validation"]

        elif stage == "validate":
            if not self.include_unknown:
                ds = ds.filter(lambda x: x["label"] != self.unseen_label_id)
            self.train = ds["train"]

        elif stage == "test":
            self.test = ds["test"]

    def train_dataloader(self):
        if self.k_1 !=False:
            print("if")
            pos_loader = DataLoader(
                self.train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=default_data_collator,
            )
            neg_loader = DataLoader(
                self.neg,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=default_data_collator,
            )
            return {"pos": pos_loader, "neg": neg_loader}
        else:
            print("else")
            loader = DataLoader(
                self.train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=default_data_collator,
            )
            return loader
            
    def val_dataloader(self):
        loader = DataLoader(
            self.validation,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=default_data_collator,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=default_data_collator,
        )
        return loader

    def preprocess_function(self, examples, label_to_id, tokenizer):
        texts = examples["text"]
        result = tokenizer(
            texts, max_length=self.max_seq_len, padding="max_length", truncation=True
        )

        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id.get(l, self.unseen_label_id)) for l in examples["label"]]

        return result


