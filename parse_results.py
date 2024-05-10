import os

def parse_finetuning():
    directory = 'ft_models/finetune_e1.0_d1e-10'

    accuracies = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        print(f)

        with open(os.path.join(f, 'train_log.txt'), 'r') as tf:
            acc_line = tf.readlines()[-1]
            if 'Test acc' in acc_line:
                acc = float(acc_line.split(' ')[-1])
                print(acc)
                accuracies.append((acc, f))

    accuracies.sort(reverse=True)
    print(accuracies)

def parse_fromscratch():
    directory = 'scratch_models/scratch_e5.0_d1e-10'

    accuracies = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        print(f)

        with open(os.path.join(f, 'train_log.txt'), 'r') as tf:
            acc_line = tf.readlines()[-1]
            if 'Test acc' in acc_line:
                acc = float(acc_line.split(' ')[-1])
                print(acc)
                accuracies.append((acc, f))

    accuracies.sort(reverse=True)
    print(accuracies)

def parse_fromscratch_nonpriv():
    directory = 'scratch_models_nonpriv/'

    accuracies = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        print(f)

        with open(os.path.join(f, 'train_log.txt'), 'r') as tf:
            acc_line = tf.readlines()[-1]
            if 'Test acc' in acc_line:
                acc = float(acc_line.split(' ')[-1])
                print(acc)
                accuracies.append((acc, f))

    accuracies.sort(reverse=True)
    print(accuracies)
    
def parse_lp():
    fname = 'lp_gridsearch.csv'
    vals = []
    with open(fname, 'r') as f:
        for l in f.readlines():
            data = [float(x) for x in l.strip().split()]
            (l, e, noise, c, accuracy, eps) = data
            vals.append((accuracy, (l, e, noise, c, accuracy, eps)))

    vals.sort(reverse=True)
    for v in vals:
        print(v)

parse_finetuning()
