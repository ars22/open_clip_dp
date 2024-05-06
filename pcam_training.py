from model_utils import *

import torch

def finetune_nonprivate(init_lr, epochs, batch, folder_prefix):
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
    
    network, preprocess = init_model(VITB, LAION) 
    (train_data, test_data) = init_pcam(root='.', preprocess=preprocess)

    lp = torch.nn.Linear(in_features=512, out_features=2)
    model = Model(network, lp).cuda()
    model, optimizer, data_loader, lr_scheduler = init_training(model, init_lr, epochs, batch, train_data)

    model = train_loop(model, optimizer, lr_scheduler, epochs, batch,
                       data_loader, folder_prefix)

    test_accuracy = eval(model, test_data, folder_prefix)

    return (model, test_accuracy)
    
def finetune_private(init_lr, epochs, batch, clip, eps, delta, folder_prefix):
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
    
    network, preprocess = init_model(VITB, LAION, private=True) 
    (train_data, test_data) = init_pcam(root='.', preprocess=preprocess)

    lp = torch.nn.Linear(in_features=512, out_features=2)
    model = Model(network, lp).cuda()
    model, optimizer, data_loader, lr_scheduler, privacy_engine = priv_init_training(model, init_lr, epochs, batch,
                                                                                     clip, eps, delta, 
                                                                                     train_data)

    model = train_loop(model, optimizer, lr_scheduler, epochs, batch,
                       data_loader, folder_prefix)

    test_accuracy = eval(model, test_data, folder_prefix)

    return (model, test_accuracy)

def train_fromscratch_nonprivate(init_lr, epochs, batch, folder_prefix):
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
    
    network, preprocess = init_model(VITB)
    (train_data, test_data) = init_pcam(root='.', preprocess=preprocess)

    lp = torch.nn.Linear(in_features=512, out_features=2)
    model = Model(network, lp).cuda()
    model.apply(initialize_weights)

    model, optimizer, data_loader, lr_scheduler = init_training(model, init_lr, epochs, batch, train_data)

    model = train_loop(model, optimizer, lr_scheduler, epochs, batch,
                       data_loader, folder_prefix)

    test_accuracy = eval(model, test_data, folder_prefix)

    return (model, test_accuracy)

def train_fromscratch_private(init_lr, epochs, batch, clip, eps, delta, folder_prefix):
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
    
    network, preprocess = init_model(VITB, pretrained_name=None, private=True) 
    (train_data, test_data) = init_pcam(root='.', preprocess=preprocess)

    lp = torch.nn.Linear(in_features=512, out_features=2)
    model = Model(network, lp).cuda()
    model.apply(initialize_weights)
    
    model, optimizer, data_loader, lr_scheduler, privacy_engine = priv_init_training(model, init_lr, epochs, batch,
                                                                                     clip, eps, delta, 
                                                                                     train_data)

    model = train_loop(model, optimizer, lr_scheduler, epochs, batch,
                       data_loader, folder_prefix)

    test_accuracy = eval(model, test_data, folder_prefix)

    return (model, test_accuracy)

def main():
    lrs = [1e-6, 1e-5, 1e-4]
    epochs=10
    batch=32
    
    eps=5.0
    delta=1e-10

    #clips=[0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0]
    clips=[1.0, 2.5, 5.0]#, 7.5, 10.0]

    folder = 'runs/private_scratch/'
    
    for lr in lrs:
        for clip in clips:
            folder += 'l{}_e{}_b{}_c{}_eps{}_del{}/'.format(lr, epochs, batch, clip, eps, delta)
            train_fromscratch_private(lr, epochs, batch, clip, eps, delta, folder)
            
if __name__=='__main__':
    main()
