import os
import yaml

method = os.environ.get('method')
model_name_or_path = os.environ.get('model_name_or_path')  # bert-base-uncased
dataset = os.environ.get('dataset')  #stackoverflow
known_cls_ratio = os.environ.get('known_cls_ratio')  # 0.5
#known_cls_ratio = float(known_cls_ratio)
max_epoch = os.environ.get('max_epoch')


# print("model", model_name_or_path)

config = {
    'data' : {
        'data_path': 'data',
        'dataset': dataset,
        'model_name_or_path': model_name_or_path,
        'known_cls_ratio': float(known_cls_ratio),
        'labeled_ratio': 1.0,
        'max_seq_len': 45,
        'batch_size': 64,
        'num_workers': 8,
        'k_1': False,
    },
    'model' : {
        'class_path': 'knncl.knncl.KNNCL',
        'init_args':{
            'model_name_or_path': model_name_or_path,
            'dropout_prob': 0.1,
            'lr': 1.0e-05,
            'weight_decay': 0.01,
            'scheduler_type': 'linear',
            'warmup_steps': 0,
            'lof_path': './lof/'+method+'_'+model_name_or_path+'_'+dataset+'_'+str(known_cls_ratio)+'_'+max_epoch+'_lof.pkl',
            'tsne_path': './tsne/'+method+'_'+model_name_or_path+'_'+dataset+'_'+str(known_cls_ratio)+'_'+max_epoch,
        }
    },
    'trainer' : {
        'accelerator': 'auto',
        'strategy': 'ddp_find_unused_parameters_true',
        'devices': [0],
        'num_nodes': 1,
        'precision': '32-true',
        'logger': None,
        'fast_dev_run': False,
        'max_epochs': int(max_epoch),
        'callbacks': [
            {
                'class_path': 'lightning.pytorch.callbacks.EarlyStopping',
                'init_args':{
                    'patience': 100,
                    'monitor': 'val_acc',
                    'mode': 'max'
                }
            },
            {
                'class_path': 'lightning.pytorch.callbacks.ModelCheckpoint',
                'init_args':{
                    'dirpath': 'outputs',
                    'filename': method+'_'+model_name_or_path+'_'+dataset+'_'+str(known_cls_ratio)+'_'+max_epoch,
                    'save_top_k': 1,
                    'monitor': 'val_acc',
                    'mode': 'max'
                }
            },
            {
                'class_path': 'lightning.pytorch.callbacks.RichProgressBar'
                }
        ],
    'min_epochs': None,
    'max_steps': -1,
    'overfit_batches': 0.0,
    'val_check_interval': None,
    'check_val_every_n_epoch': 1,
    'num_sanity_val_steps': None,
    'log_every_n_steps': None,
    'enable_checkpointing': None,
    'enable_progress_bar': None,
    'enable_model_summary': None,
    'accumulate_grad_batches': 1,
    'gradient_clip_val': 0.25,
    'gradient_clip_algorithm': None,
    'deterministic': None,
    'benchmark': None,
    'inference_mode': True,
    'use_distributed_sampler': True,
    'profiler': None,
    'detect_anomaly': False,
    'barebones': False,
    'plugins': None,
    'sync_batchnorm': False,
    'reload_dataloaders_every_n_epochs': 0,
    'default_root_dir': None
    },
    'project' : {
        'path' : './csv/KNN_'+model_name_or_path+'_'+dataset+'_'+str(known_cls_ratio)+'_'+max_epoch+'.csv'
    }
}

filename = f"/workspace/openlib/samples/{method}_{model_name_or_path}_{dataset}_{known_cls_ratio}_{max_epoch}_output.yaml"
with open(filename, 'w') as file:
    yaml.dump(config, file)

