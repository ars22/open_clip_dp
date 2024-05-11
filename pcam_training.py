from model_utils import *

import torch

def finetune_nonprivate(init_lr, epochs, batch, folder_prefix, subsample=-1):
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
    
    network, preprocess = init_model(VITB, LAION) 
    (train_data, test_data) = init_pcam(root='.', preprocess=preprocess, subsample=subsample)

    lp = torch.nn.Linear(in_features=512, out_features=2)
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

    lp = torch.nn.Linear(in_features=512, out_features=2)
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

    lp = torch.nn.Linear(in_features=512, out_features=2)
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

    lp = torch.nn.Linear(in_features=512, out_features=2)
    model = Model(network, lp).cuda()
    model.apply(initialize_weights)
    
    model, optimizer, data_loader, lr_scheduler, privacy_engine = priv_init_training(model, init_lr, epochs, batch,
                                                                                     clip, eps, delta, 
                                                                                     train_data)

    model = train_loop(model, optimizer, lr_scheduler, epochs, batch,
                       data_loader, folder_prefix)

    test_accuracy = eval(model, test_data, batch, folder_prefix)

    return (model, test_accuracy)

def main():
    #lrs = [1e-6, 1e-5, 1e-4]
    lrs = [3e-3]
    epochs=20
    batch=128
    
    epsvals=[0.3, 0.4]
    delta=1e-10

    #clips=[0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0]
    #clips=[1.0, 2.5, 5.0]#, 7.5, 10.0]
    clips = [1.0]

    subsamples = [10000, -1]
    
    folder = 'runs/nonprivate_fromscratch/'

    for subsample in subsamples:
        for lr in lrs:
            savefolder = folder + 'l{}_e{}_b{}_subs{}/'.format(lr, epochs, batch, subsample)
            print('>>>>>>', savefolder)
            (model, acc) = train_fromscratch_nonprivate(lr, epochs, batch, savefolder, subsample)
                    
if __name__=='__main__':
    main()
