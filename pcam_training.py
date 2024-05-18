from model_utils import *

import torch

import typer

def finetune_nonprivate(init_lr, epochs, batch, folder_prefix, subsample=-1):
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
    
    network, preprocess = init_model(VITB, LAION) 
    (train_data, test_data) = init_pcam(root='.', preprocess=preprocess, subsample=subsample)

    lp = torch.nn.Linear(in_features=512, out_features=PCAM_LABELS)
    model = Model(network, lp).cuda()
    model, optimizer, data_loader, lr_scheduler = init_training(model, init_lr, epochs, batch, train_data)

    model = train_loop(model, optimizer, lr_scheduler, epochs, batch,
                       data_loader, folder_prefix)

    test_accuracy = eval(model, test_data, batch, folder_prefix)

    return (model, test_accuracy)
    
def finetune_private(init_lr, epochs, batch, clip, eps, delta, folder_prefix, subsample=-1):
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
    
    network, preprocess = init_model(VITB, LAION, private=True) 
    (train_data, test_data) = init_pcam(root='.', preprocess=preprocess, subsample=subsample)

    lp = torch.nn.Linear(in_features=512, out_features=PCAM_LABELS)
    model = Model(network, lp).cuda()
    model, optimizer, data_loader, lr_scheduler, privacy_engine = priv_init_training(model, init_lr, epochs, batch,
                                                                                     clip, eps, delta, 
                                                                                     train_data)

    model = train_loop(model, optimizer, lr_scheduler, epochs, batch,
                       data_loader, folder_prefix)

    test_accuracy = eval(model, test_data, batch, folder_prefix)

    return (model, test_accuracy)

def train_fromscratch_nonprivate(init_lr, epochs, batch, folder_prefix, subsample=-1):
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
    
    network, preprocess = init_model(VITB)
    (train_data, test_data) = init_pcam(root='.', preprocess=preprocess, subsample=subsample)

    lp = torch.nn.Linear(in_features=512, out_features=PCAM_LABELS)
    model = Model(network, lp).cuda()
    model.apply(initialize_weights)

    model, optimizer, data_loader, lr_scheduler = init_training(model, init_lr, epochs, batch, train_data)

    model = train_loop(model, optimizer, lr_scheduler, epochs, batch,
                       data_loader, folder_prefix)

    test_accuracy = eval(model, test_data, batch, folder_prefix)

    return (model, test_accuracy)

def train_fromscratch_private(init_lr, epochs, batch, clip, eps, delta, folder_prefix, subsample=-1):
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
    
    network, preprocess = init_model(VITB, pretrained_name=None, private=True) 
    (train_data, test_data) = init_pcam(root='.', preprocess=preprocess, subsample=subsample)

    lp = torch.nn.Linear(in_features=512, out_features=PCAM_LABELS)
    model = Model(network, lp).cuda()
    model.apply(initialize_weights)
    
    model, optimizer, data_loader, lr_scheduler, privacy_engine = priv_init_training(model, init_lr, epochs, batch,
                                                                                     clip, eps, delta, 
                                                                                     train_data)

    model = train_loop(model, optimizer, lr_scheduler, epochs, batch,
                       data_loader, folder_prefix)

    test_accuracy = eval(model, test_data, batch, folder_prefix)

    return (model, test_accuracy)

def main(lr: float = typer.Option(default=...),
         epochs: int = typer.Option(default=...),
         batch: int = typer.Option(default=32),
         eps: float = typer.Option(default=None),
         delta: float = typer.Option(default=1e-10),
         clip: float  = typer.Option(default=None),
         subsample: bool = typer.Option(default=False)
         ):
   # lrs = [5e-5, 1e-4]
   # epochs=8
   # batch=32
    
   # eps=0.3
   # delta=1e-10

    #clips=[0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0]
   # clips=[2.]#, 7.5, 10.0]

    folder = 'runs_resisc/private_scratch/'

    if subsample:
        subsample = 10000
    else:
        subsample = -1
    
    if eps is None:
        print('Nonprivate')
        folder_ = folder + 'l{}_e{}_b{}/'.format(lr, epochs, batch)
        print(folder_)
        train_fromscratch_nonprivate(lr, epochs, batch, folder_, subsample)

    else:
        folder_ = folder + 'l{}_e{}_b{}_c{}_eps{}_del{}/'.format(lr, epochs, batch, clip, eps, delta)
        print(folder_)
        print('Private finetuning, subsample = ', subsample)
        finetune_private(lr, epochs, batch, clip, eps, delta, folder_, subsample)
        #train_fromscratch_nonprivate(lr, epochs, batch, folder)
            
if __name__=='__main__':
    typer.run(main)
