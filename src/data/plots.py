import matplotlib.pyplot as plt
import nibabel as nib
import json
import numpy as np
import random
from sklearn import metrics


def load_tensors_dump(run):
    run_file = '/local/dhaziza/data/%s/tensors_dump.json' % run
    dump_tensors = json.load(open(run_file, 'r'))
    return {
        int(k): v
        for k, v in dump_tensors.items()
    }

def load_datasets(run='20180411-120122', global_step_idx=-1, dump_tensors=None, num_classes=3, valid_ratio=0.2):
    if dump_tensors is None:
        dump_tensors = load_tensors_dump(run)
    global_steps = sorted(dump_tensors.keys())
    step = global_steps[global_step_idx]
    #print('Analysis on step %d' % step)
    data_per_class_train = {0: [], 1: [], 2: []}
    data_per_class_valid = {0: [], 1: [], 2: []}
    data_per_class_all = {0: [], 1: [], 2: []}
    p_id_to_dataset = {}
    for output in dump_tensors[step]:
        for label, proba, p_id in zip(output['classifier/labels'], output['classifier/logits'], output['classifier/study_patient_id']):
            #proba[1] = -proba[0]
            label_idx = np.argmax(label)
            p_id = p_id[0]
            # p_id = random.randint(0, 100000)
            if p_id not in p_id_to_dataset:
                if random.random() < valid_ratio:
                    p_id_to_dataset[p_id] = 'valid'
                else:
                    p_id_to_dataset[p_id] = 'train'
            if p_id_to_dataset[p_id] == 'valid':
                data_per_class_valid[label_idx].append(proba)
            else:
                data_per_class_train[label_idx].append(proba)
            data_per_class_all[label_idx].append(proba)
    for k in range(num_classes):
        data_per_class_valid[k] = np.array(data_per_class_valid[k])
        data_per_class_train[k] = np.array(data_per_class_train[k])
        data_per_class_all[k] = np.array(data_per_class_all[k])
    return data_per_class_train, data_per_class_valid, data_per_class_all

def load_smt_outcome(run):
    run_file = '/local/dhaziza/data/%s/sumatra_outcome.json' % run
    return json.load(open(run_file, 'r'))

def load_smt_run(run):
    outcome = load_smt_outcome(run)
    return outcome['text_outcome'], outcome['run_reason']
    
def print_class_samples(dataset, name):
    print('Dataset: %s' % name)
    for k, v in dataset.items():
        print('  Class %d: %d items' % (k, len(v)))

def plot_smt_metric(reason_to_runs, metrics, xlabel=None, ylabel='Accuracy'):
    if not isinstance(metrics, list):
        metrics = [metrics]
    reasons, reason_runs = zip(*sorted(zip(reason_to_runs.keys(), reason_to_runs.values())))
    for reason, runs in zip(reasons, reason_runs):
        if not isinstance(runs, list):
            runs = [runs]
        x = []
        y = []
        for run in runs:
            smt_outcome = load_smt_outcome(run)['numeric_outcome']
            for m in metrics:
                x.append(smt_outcome[m]['x'])
                y.append(smt_outcome[m]['y'])
                if xlabel is None:
                    xlabel = smt_outcome[m]['x_label']
        plt.plot(x[0], np.mean(y, axis=0), label='%s (%s)' % (reason, m), marker="*")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()


# Train scikit-learn trees
def dataset_to_scikit(dataset):
    X = []
    Y = []
    w = []
    for k, v in dataset.items():
        if len(v) > 0:
            count = v.shape[0]
            X += v.tolist()
            Y += ([k] * count)
            w += ([1.0/count] * count)
    return X, Y, w

def check_classifier_performance(classifier, run):
    valid_scores = []
    dump_tensors = load_tensors_dump(run)
    for step in range(-7, -1):
        this_step_scores = []
        for i in range(100):
            dataset_train, dataset_valid, _ = load_datasets(run, step, dump_tensors=dump_tensors)
            scikit_train_x, scikit_train_y, train_w = dataset_to_scikit(dataset_train)
            scikit_valid_x, scikit_valid_y, valid_w = dataset_to_scikit(dataset_valid)
            classifier = classifier.fit(scikit_train_x, scikit_train_y, train_w)
            this_step_scores.append(classifier.score(scikit_valid_x, scikit_valid_y, valid_w))
        valid_scores.append(np.mean(this_step_scores))
    print("  Valid score: %.3f [%.2f - %.2f]" % (
        np.mean(valid_scores),
        np.percentile(valid_scores, 5),
        np.percentile(valid_scores, 95),
    ))

    def predict_classifier_decision(x):
        return classifier.predict([x])
    #check_accuracy(dataset_valid, predict_classifier_decision)
    return valid_scores

def do_bars_comparison_plot(runs_reasons, series, x_label='', title='untitled', ylim=[0.7, 0.9],
                           baseline_colors=None,
                           colors=None,
                           classifier=None,
):
    type_to_color = {
        # Run, Baseline
        '3DCNN': ['xkcd:light blue', 'xkcd:bright blue'],
        '3DCNN_BN': ['xkcd:light blue', 'xkcd:bright blue'],
        '3DCNN_NOBN': ['xkcd:sand', 'xkcd:mustard'],
        'Inceptionv3': ['xkcd:sand', 'xkcd:mustard'],
        'SVM': ['xkcd:light green', 'xkcd:bright green'],
    }
    type_to_color = {
        # Run, Baseline
        '3DCNN': ['xkcd:green', 'xkcd:blue'],
        '3DCNN_BN': ['xkcd:green', 'xkcd:blue'],
        '3DCNN_NOBN': ['xkcd:forest green', 'xkcd:royal blue'],
        'Inceptionv3': ['xkcd:olive drab', 'xkcd:grape'],
        'SVM': ['xkcd:orange', 'xkcd:red'],
    }
    if baseline_colors is None:
        baseline_colors = [type_to_color[t][1] for t in series]
    if colors is None:
        colors = [type_to_color[t][0] for t in series]
    reasons, runs = zip(*sorted(zip(runs_reasons.keys(), runs_reasons.values())))
    series = [{
            'values': [],
            'mins': [],
            'maxs': [],
            'name': sname,
            'baseline_color': bc,
            'color': c,
        }
        for sname, bc, c in zip(series, baseline_colors, colors)
    ]
    for reason_runs in runs:
        for serie_id, serie in enumerate(series):
            try:
                repeat_runs = reason_runs[serie_id]
                if not isinstance(repeat_runs, list):
                    repeat_runs = [repeat_runs]
                v = []
                for r in repeat_runs:
                    try:
                        outcome, reason = load_smt_run(r)
                        print('Run %s (%s)' % (r, reason))
                        print('  ## %s' % outcome)
                        assert(outcome != 'TODO')
                        if classifier is None:
                            v += [float(outcome.split(' ')[3])]
                        else:
                            v += [np.mean(check_classifier_performance(classifier, r))]
                    except Exception as e:
                        raise e
                assert(len(v) > 0)
                serie['values'].append(np.median(v))
                #serie['values'].append(float(outcome.split(' ')[3]))
                serie['mins'].append(np.percentile(v, 5))
                serie['maxs'].append(np.percentile(v, 95))
            except Exception as e:
                serie['values'].append(0)
                serie['mins'].append(0)
                serie['maxs'].append(0)
                raise e
                

    barWidth = 0.2
    r0 = np.arange(len(runs))

    # Create blue bars
    for serie_id, serie in enumerate(series):
        r_pos = [x + serie_id * barWidth for x in r0]
        yerr = [
            np.array(serie['values']) - np.array(serie['mins']),
            np.array(serie['maxs']) - np.array(serie['values']),
        ]
        color = [
            serie['baseline_color'] if 'baseline' in reasons[i] else serie['color']
            for i in range(len(reasons))
        ]
        plt.bar(r_pos, serie['values'], width = barWidth, yerr=yerr, capsize=7, label=serie['name'], color=color)

    # general layout
    plt.xticks([
        r + (0.5 * len(series) - 0.5) * barWidth
        for r in r0
    ], [r.split('__')[-1] for r in reasons])
    plt.ylim(ylim[0], ylim[1])
    plt.ylabel('AD/HC validation accuracy')
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()

    # Show graphic
    plt.show()


def display_image_path(path):
    img = nib.load(path)
    dat = img.get_data()
    if len(dat.shape) == 4:
        dat = dat[:,:,:, 0]
    imshowargs = {
        'interpolation': 'nearest',
    }

    def forceAspect(ax, aspect=1):
        im = ax.get_images()
        extent =  im[0].get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

    ax = plt.subplot(131)
    ax.imshow(dat[dat.shape[0]/2,:,:], **imshowargs)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    forceAspect(ax)
    
    ax = plt.subplot(132)
    ax.imshow(dat[:,dat.shape[1]/2,:], **imshowargs)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    forceAspect(ax)
    
    ax = plt.subplot(133)
    ax.imshow(dat[:,:, dat.shape[2]/2], **imshowargs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    forceAspect(ax)
    
    plt.tight_layout()
    plt.show()
    return dat