"""
Description     : Simple Python implementation of the Apriori Algorithm
Modified from:  https://github.com/asaini/Apriori
Usage:
    $python apriori.py -f DATASET.csv -s minSupport

    $python apriori.py -f DATASET.csv -s 0.15
"""

import os
import sys
import time

from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser


def subsets(arr):
    """Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)

    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet


# 得到itemset的組合
def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    _joinSet = set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )
    numBefore = len(_joinSet)

    # Pruning
    _joinSet = set()
    for i in itemSet:
        for j in itemSet:
            if len(i.union(j)) == length:
                # Find the subsets with len == length-1
                _subsets = [
                    frozenset(subset)
                    for subset in subsets(i.union(j))
                    if len(frozenset(subset)) == length - 1
                ]
                # Check whether all the subsets is in candidate
                if all((subset in itemSet) for subset in _subsets):
                    _joinSet.add(i.union(j))

    numAfter = len(_joinSet)

    return _joinSet, [numBefore, numAfter]


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets

    return itemSet, transactionList


def runApriori(data_iter, minSupport):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()
    pruningList = list()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)

    currentLSet = oneCSet
    pruningList.append([len(currentLSet), len(currentLSet)])
    k = 2
    while currentLSet != set([]):
        largeSet[k - 1] = currentLSet
        currentLSet, numPruning = joinSet(currentLSet, k)
        pruningList.append(numPruning)
        currentCSet = returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet
        )
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    return toRetItems, pruningList


def printResults(items):
    """prints the generated itemsets sorted by support"""
    for item, support in sorted(items, key=lambda x: x[1]):
        print("item: %s , %.3f" % (str(item), support))


def to_str_results(items):
    """prints the generated itemsets sorted by support"""
    i = []
    for item, support in sorted(items, key=lambda x: x[1]):
        x = "item: %s , %.3f" % (str(item), support)
        i.append(x)
    return i


def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    with open(fname, "r") as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(",")  # Remove trailing comma
            record = frozenset(line.split(","))
            yield record


if __name__ == "__main__":

    optparser = OptionParser()
    optparser.add_option(
        "-f",
        "--inputFile",
        dest="input",
        help="filename containing csv",
        default="A.csv",
    )
    optparser.add_option(
        "-o",
        "--outputPath",
        dest="output",
        help="output folder path",
        default="output/Apriori/",
    )
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.1,
        type="float",
    )

    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
        inFile = sys.stdin
    elif options.input is not None:
        inFile = dataFromFile(options.input)
        filename = options.input.replace(".csv", "")
    else:
        print("No dataset filename specified, system with exit\n")
        sys.exit("System will exit")

    minSupport = options.minS

    os.makedirs(f"{options.output}", exist_ok=True)

    # ============ Task 1 ============
    task1_start = time.time()
    items, pruningList = runApriori(inFile, minSupport)

    # ============ (a) ============
    # Print out all frequent itemsets with support
    with open(
        f"{options.output}step2_task1_dataset({filename})_{options.minS}_result1.txt",
        "w",
    ) as file:
        for item, support in sorted(items, key=lambda x: x[1], reverse=True):
            file.write(f"{support*100 :.1f}%\t{set(item)}\n")

    # ============ (b) ============
    with open(
        f"{options.output}step2_task1_dataset({filename})_{options.minS}_result2.txt",
        "w",
    ) as file:
        # Print out the total number of frequent itemsets
        file.write(f"{len(items)}\n")
        # Print out the number of candidates generated before and after pruning
        for i, numPruning in enumerate(pruningList):
            file.write(f"{i+1}\t{numPruning[0]}\t{numPruning[1]}\n")

    task1_end = time.time()
    print(f"Task1 computation time: {task1_end - task1_start}")

    # ============ Task 2 ============
    for item, support in sorted(items, key=lambda x: len(x[0]), reverse=True):
        _subsets = set(subsets(item))
        # Remove itself in subsets
        _subsets.remove(item)
        for subset in _subsets:
            for i, (itemSubset, supportSubset) in enumerate(items):
                # Check whether the support of subsets == the support of itemsets
                if (
                    frozenset(subset) == frozenset(itemSubset)
                    and support == supportSubset
                ):
                    items.pop(i)

    with open(
        f"{options.output}step2_task2_dataset({filename})_{options.minS}_result1.txt",
        "w",
    ) as file:
        # Print out the total number of frequent closed itemset
        file.write(f"{len(items)}\n")
        # Print out all frequent closed itemset with support
        for item, support in sorted(items, key=lambda x: x[1], reverse=True):
            file.write(f"{support*100 :.1f}%\t{set(item)}\n")

    task2_end = time.time()

    print(f"Task2 computation time: {task2_end - task1_start}")
    print(f"Task2/Task1: {(task2_end - task1_start)/(task1_end - task1_start)*100}%")
