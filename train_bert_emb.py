import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from bert_emb import ProtBertBFDClassifier
from data_utils import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser = ProtBertBFDClassifier.add_model_specific_args(parser)
# parser = pl.Trainer.add_argparse_args(parser)

# Add trainer-specific arguments manually
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--devices', type=int, nargs='+', default=[0])
parser.add_argument('--accelerator', type=str, default='auto')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--val_batch_size', type=int, default=24)

hparams = parser.parse_args()

print("got hparams", hparams)

if __name__ == "__main__":
    train_path = "/Users/dkim0/Downloads/Ancestor_Sequence_Augmentation/go_metric/go_bench"
    train_dataset = BertSeqDataset.from_memory(f"{train_path}/training_molecular_function_annotations.tsv", 
                                               f"{train_path}/molecular_function_terms.json", 
                                               f"{train_path}/../uniprot_reviewed.fasta", cache_dir='/Users/dkim0/Downloads/Ancestor_Sequence_Augmentation/go_metric/cache')
    val_dataset = BertSeqDataset.from_memory(f"{train_path}/validation_molecular_function_annotations.tsv", 
                                               f"{train_path}/molecular_function_terms.json", 
                                               f"{train_path}/../uniprot_reviewed.fasta", cache_dir='/Users/dkim0/Downloads/Ancestor_Sequence_Augmentation/go_metric/cache')

    collate_seqs = get_bert_seq_collator(max_length=hparams.max_length, add_special_tokens=True)
    dataloader_params = {"shuffle": True, "batch_size": 8, "collate_fn":collate_seqs}
    val_dataloader_params = {"shuffle": False, "batch_size": 24, "collate_fn":collate_seqs}

    train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)

    hparams.num_classes = train_dataset[0]['labels'].shape[0]

    model = ProtBertBFDClassifier(hparams)
    
    early_stop_callback = EarlyStopping(monitor='F1/val', min_delta=0.00, patience=3, verbose=True, mode='max')
    checkpoint_callback = ModelCheckpoint(filename="/Users/dkim0/Downloads/Ancestor_Sequence_Augmentation/go_metric/checkpoints/bert_emb_sample", 
                                          verbose=True, monitor='F1/val', mode='max')
    # trainer = pl.Trainer.from_argparse_args(hparams, accelerator='gpu', devices=[1], max_epochs=100, profiler='simple',
    #                                          callbacks=[early_stop_callback, checkpoint_callback])
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger("logs", name="bert_emb")
    # trainer = pl.Trainer(devices=[0], max_epochs=150, 
    #                      callbacks=[early_stop_callback, checkpoint_callback], logger=logger)
    trainer = pl.Trainer(
        devices=hparams.devices,
        max_epochs=hparams.max_epochs,
        accelerator=hparams.accelerator,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger
    )
    trainer.fit(model, train_loader, val_loader)