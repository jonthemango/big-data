#!/usr/bin/env python3
# (C) 2017 OpenEye Scientific Software Inc. All rights reserved.
#
# TERMS FOR USE OF SAMPLE CODE The software below ("Sample Code") is
# provided to current licensees or subscribers of OpenEye products or
# SaaS offerings (each a "Customer").
# Customer is hereby permitted to use, copy, and modify the Sample Code,
# subject to these terms. OpenEye claims no rights to Customer's
# modifications. Modification of Sample Code is at Customer's sole and
# exclusive risk. Sample Code may require Customer to have a then
# current license or subscription to the applicable OpenEye offering.
# THE SAMPLE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED.  OPENEYE DISCLAIMS ALL WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. In no event shall OpenEye be
# liable for any damages or liability in connection with the Sample Code
# or its use.

###########
# you can run it using
# python roc_curves.py actives.txt scores.txt roc.png
#############################################################################
# Plots ROC curve
#############################################################################

import sys
import os
from operator import itemgetter
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def main(argv=[__name__]):

    if len(sys.argv) != 4:
        print("Usage: <actives> <scores> <image>")
        return 1

    afname = sys.argv[1]
    sfname = sys.argv[2]
    ofname = sys.argv[3]

    # f, ext = os.path.splitext(ofname)
    # if not is_supported_image_type(ext):
    #     print("Format \"%s\" is not supported!" % ext)
    #     return 1

    # read id of actives

    actives = load_actives(afname)
    print("Loaded %d actives from %s" % (len(actives), afname))

    # read molecule id - score pairs

    label, scores = load_scores(sfname)
    print("Loaded %d %s scores from %s" % (len(scores), label, sfname))

    # sort scores by ascending order
    sortedscores = sorted(scores, key=itemgetter(1))

    print("Plotting ROC Curve ...")
    color = "#008000"  # dark green
    depict_ROC_curve(actives, sortedscores, label, color, ofname)

    return 0


def load_actives(fname):

    actives = []
    for line in open(fname, 'r').readlines():
        id = line.strip()
        actives.append(id)

    return actives


def load_scores(fname):

    sfile = open(fname, 'r')
    label = sfile.readline()
    label = label.strip()

    scores = []
    for line in sfile.readlines():
        id, score = line.strip().split()
        scores.append((id, float(score)))

    return label, scores


def get_rates(actives, scores):
    """
    :type actives: list[sting]
    :type scores: list[tuple(string, float)]
    :rtype: tuple(list[float], list[float])
    """

    tpr = [0.0]  # true positive rate
    fpr = [0.0]  # false positive rate
    nractives = len(actives)
    nrdecoys = len(scores) - len(actives)

    foundactives = 0.0
    founddecoys = 0.0
    for idx, (id, score) in enumerate(scores):
        if id in actives:
            foundactives += 1.0
        else:
            founddecoys += 1.0

        tpr.append(foundactives / float(nractives))
        fpr.append(founddecoys / float(nrdecoys))

    return tpr, fpr


def setup_ROC_curve_plot(plt):
    """
    :type plt: matplotlib.pyplot
    """

    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)


def save_ROC_curve_plot(plt, filename, randomline=True):
    """
    :type plt: matplotlib.pyplot
    :type fname: string
    :type randomline: boolean
    """

    if randomline:
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(filename)


def depict_ROC_curve(actives, scores, label, color, filename, randomline=True):
    """
    :type actives: list[sting]
    :type scores: list[tuple(string, float)]
    :type color: string (hex color code)
    :type fname: string
    :type randomline: boolean
    """

    plt.figure(figsize=(4, 4), dpi=80)

    setup_ROC_curve_plot(plt)
    add_ROC_curve(plt, actives, scores, color, label)
    save_ROC_curve_plot(plt, filename, randomline)


def add_ROC_curve(plt, actives, scores, color, label):
    """
    :type plt: matplotlib.pyplot
    :type actives: list[sting]
    :type scores: list[tuple(string, float)]
    :type color: string (hex color code)
    :type label: string
    """

    tpr, fpr = get_rates(actives, scores)
    roc_auc = auc(fpr, tpr)

    roc_label = '{} (AUC={:.3f})'.format(label, roc_auc)
    plt.plot(fpr, tpr, color=color, linewidth=2, label=roc_label)


def is_supported_image_type(ext):
    fig = plt.figure()
    return (ext[1:] in fig.canvas.get_supported_filetypes())


if __name__ == "__main__":
    sys.exit(main(sys.argv))
