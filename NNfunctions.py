# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 18:45:36 2018

@author: Kel3vra
"""
def permutated(x_train,x_test):
    import numpy as np
    rng_permute = np.random.RandomState(1)
    idx_permute = rng_permute.permutation(784)
    # =============================================================================
    x_train=x_train[:,idx_permute]
    x_test =x_test[:,idx_permute]
    return x_train,x_test
    
# =============================================================================
# Confusion matrix plot
# =============================================================================
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6),tight_layout=True)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}  misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('normalized confusion matrix.png')
    plt.show()
    
def plot_confusion_matrix_train(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6),tight_layout=True)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}  misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('normalized confusion matrix train.png')
    plt.show()
    
def heatmap(confusion_mlp_RMSprop):
    import seaborn as sn
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    fig2=plt.figure()
    fig2.add_subplot(111)
    sn.heatmap(confusion_mlp_RMSprop,annot=True,square=True,cbar=False,fmt="d",annot_kws={"size": 9})
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    plt.savefig('Confusion_matrix_mlp.png')  
    plt.show() 
    
    
def heatmap_train(confusion_mlp_RMS):
    import seaborn as sn
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    fig2=plt.figure()
    fig2.add_subplot(111)
    sn.heatmap(confusion_mlp_RMS,annot=True,square=True,cbar=False,fmt="d",annot_kws={"size": 9})
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    plt.savefig('Confusion_matrix_mlp_train.png')  
    plt.show() 
def loss(history):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    fig3=plt.figure(tight_layout=True)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epochs\nloss train_set={:0.4f}  loss test_set={:0.4f}'.format(history.history['loss'][-1], history.history['val_loss'][-1]))
    plt.legend(['loss_train', 'loss_test'])
    plt.savefig('loss_error.png')
    plt.show()

def accuracy(history):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    fig4=plt.figure(tight_layout=True)
    accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs\naccuracy train_set={:0.4f}  accuracy test_set={:0.4f}'.format(history.history['acc'][-1], history.history['val_acc'][-1]))
    plt.legend(['acc_train', 'acc_test'])
    plt.savefig('accuracy.png')
    plt.show()

def accuracy2(history,i):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    fig4=plt.figure(tight_layout=True)
    accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs\naccuracy train_set={:0.4f}  accuracy test_set={:0.4f}'.format(history.history['acc'][-1], history.history['val_acc'][-1]))
    plt.legend(['acc_train', 'acc_test'])
    plt.savefig('accuracy{}.png'.format(i))
    plt.show()

def unique_test(y_test_class):
    from collections import Counter
    #count the numbers of unique numbers in test
    y_uni = list(Counter(y_test_class).items())
    y_uni = dict((y, x) for x, y in y_uni)
    return y_uni


def unique_train(y_train_class):
    from collections import Counter
    #count the numbers of unique numbers in train
    y_uni_train = list(Counter(y_train_class).items())
    y_uni_train = dict((y, x) for x, y in y_uni_train)
    return y_uni_train

def unique(y_train_class,y_test_class):
    from collections import Counter
    import numpy as np
    d_uni = np.hstack((y_train_class,y_test_class)).reshape(70000, 1)
    d_uni = list(Counter(d_uni).items())
    return d_uni

def plot_numbers(y_uni):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    #barplot 
    values = list(y_uni.keys())
    names = list(y_uni.values())
    fig, ax = plt.subplots()    
    width = 0.75 # the width of the bars 
    ind = np.arange(len(values))  # the x locations for the groups
    ax.barh(ind, values, width, color="brown")
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(names, minor=False)
    for i, v in enumerate(values):
        ax.text(v + 2, i + .16, str(v), color='red', fontweight='bold')
    plt.title('title')
    plt.xlabel('number of digits')
    plt.ylabel('labels')      
    #plt.show()
    plt.savefig(os.path.join('test1.png'), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures



