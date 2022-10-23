import os
import time

from collections import defaultdict
from csv import reader
from itertools import chain, combinations
from optparse import OptionParser


class Node:
    def __init__(self, itemName, frequency, parentNode):
        self.itemName = itemName
        self.count = frequency
        self.parent = parentNode
        self.children = {}
        self.next = None

    def increment(self, frequency):
        self.count += frequency

    def display(self, ind=1):
        print("  " * ind, self.itemName, " ", self.count)
        for child in list(self.children.values()):
            child.display(ind + 1)


def getFromFile(fname):
    itemSetList = []
    frequency = []

    with open(fname, "r") as file:
        csv_reader = reader(file)
        for line in csv_reader:
            line = list(filter(None, line))
            itemSetList.append(line)
            frequency.append(1)

    return itemSetList, frequency


def constructTree(itemSetList, frequency, minSup):
    headerTable = defaultdict(int)
    # Counting frequency and create header table
    for idx, itemSet in enumerate(itemSetList):
        for item in itemSet:
            headerTable[item] += frequency[idx]

    # Deleting items below minSup
    headerTable = dict(
        (item, sup) for item, sup in headerTable.items() if sup >= minSup
    )
    if len(headerTable) == 0:
        return None, None

    # HeaderTable column [Item: [frequency, headNode]]
    for item in headerTable:
        headerTable[item] = [headerTable[item], None]

    # Init Null head node
    fpTree = Node("Null", 1, None)
    # Update FP tree for each cleaned and sorted itemSet
    for idx, itemSet in enumerate(itemSetList):
        itemSet = [item for item in itemSet if item in headerTable]
        itemSet.sort(key=lambda item: headerTable[item][0], reverse=True)
        # Traverse from root to leaf, update tree with given item
        currentNode = fpTree
        for item in itemSet:
            currentNode = updateTree(item, currentNode, headerTable, frequency[idx])

    return fpTree, headerTable


def updateHeaderTable(item, targetNode, headerTable):
    if headerTable[item][1] == None:
        headerTable[item][1] = targetNode
    else:
        currentNode = headerTable[item][1]
        # Traverse to the last node then link it to the target
        while currentNode.next != None:
            currentNode = currentNode.next
        currentNode.next = targetNode


def updateTree(item, treeNode, headerTable, frequency):
    if item in treeNode.children:
        # If the item already exists, increment the count
        treeNode.children[item].increment(frequency)
    else:
        # Create a new branch
        newItemNode = Node(item, frequency, treeNode)
        treeNode.children[item] = newItemNode
        # Link the new branch to header table
        updateHeaderTable(item, newItemNode, headerTable)

    return treeNode.children[item]


def ascendFPtree(node, prefixPath):
    if node.parent != None:
        prefixPath.append(node.itemName)
        ascendFPtree(node.parent, prefixPath)


def findPrefixPath(basePat, headerTable):
    # First node in linked list
    treeNode = headerTable[basePat][1]
    condPats = []
    frequency = []
    while treeNode != None:
        prefixPath = []
        # From leaf node all the way to root
        ascendFPtree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            # Storing the prefix path and it's corresponding count
            condPats.append(prefixPath[1:])
            frequency.append(treeNode.count)

        # Go to next node
        treeNode = treeNode.next
    return condPats, frequency


def mineTree(headerTable, minSup, preFix, freqItemList):
    # Sort the items with frequency and create a list
    sortedItemList = [
        item[0] for item in sorted(list(headerTable.items()), key=lambda p: p[1][0])
    ]
    # Start with the lowest frequency
    for item in sortedItemList:
        # Pattern growth is achieved by the concatenation of suffix pattern with frequent patterns generated from conditional FP-tree
        newFreqSet = preFix.copy()
        newFreqSet.add(item)
        freqItemList.append(newFreqSet)
        # Find all prefix path, constrcut conditional pattern base
        conditionalPattBase, frequency = findPrefixPath(item, headerTable)
        # Construct conditonal FP Tree with conditional pattern base
        conditionalTree, newHeaderTable = constructTree(
            conditionalPattBase, frequency, minSup
        )
        if newHeaderTable != None:
            # Mining recursively on the tree
            mineTree(newHeaderTable, minSup, newFreqSet, freqItemList)


def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


def getSupport(testSet, itemSetList):
    count = 0
    for itemSet in itemSetList:
        if set(testSet).issubset(itemSet):
            count += 1
    return count


def associationRule(freqItemSet, itemSetList, minConf):
    rules = []
    for itemSet in freqItemSet:
        subsets = powerset(itemSet)
        itemSetSup = getSupport(itemSet, itemSetList)
        for s in subsets:
            confidence = float(itemSetSup / getSupport(s, itemSetList))
            if confidence > minConf:
                rules.append([set(s), set(itemSet.difference(s)), confidence])
    return rules


def getFrequencyFromList(itemSetList):
    frequency = [1 for i in range(len(itemSetList))]
    return frequency


def fpgrowth(itemSetList, minSupRatio, minConf):
    frequency = getFrequencyFromList(itemSetList)
    minSup = len(itemSetList) * minSupRatio
    fpTree, headerTable = constructTree(itemSetList, frequency, minSup)
    if fpTree == None:
        print("No frequent item set")
    else:
        freqItems = []
        mineTree(headerTable, minSup, set(), freqItems)
        rules = associationRule(freqItems, itemSetList, minConf)
        return freqItems, rules


def fpgrowthFromFile(fname, minSupRatio, minConf):
    itemSetList, frequency = getFromFile(fname)
    minSup = len(itemSetList) * minSupRatio
    fpTree, headerTable = constructTree(itemSetList, frequency, minSup)
    if fpTree == None:
        print("No frequent item set")
    else:
        freqItems = []
        mineTree(headerTable, minSup, set(), freqItems)
        rules = associationRule(freqItems, itemSetList, minConf)
        return freqItems, rules, itemSetList


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
        default="output/FP-Growth/",
    )
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minSup",
        help="minimum support value",
        default=0.1,
        type="float",
    )
    optparser.add_option(
        "-c",
        "--minConfidence",
        dest="minConf",
        help="minimum confidence value",
        default=0.0,
        type="float",
    )

    (options, args) = optparser.parse_args()

    filename = options.input.replace(".csv", "")
    os.makedirs(f"{options.output}", exist_ok=True)

    # ============ Task 1 ============
    task1_start = time.time()
    freqItemSet, rules, itemSetList = fpgrowthFromFile(
        options.input, options.minSup, options.minConf
    )

    # ============ (a) ============
    items = list()
    for itemSet in freqItemSet:
        itemSetSup = getSupport(itemSet, itemSetList)
        support = itemSetSup / len(itemSetList)
        if support >= options.minSup:
            items.append([itemSet, support])
    with open(
        f"{options.output}step3_task1_dataset({filename})_{options.minSup}_result1.txt",
        "w",
    ) as file:
        for itemSet, support in sorted(items, key=lambda x: x[1], reverse=True):
            file.write(f"{support*100 :.1f}%\t{itemSet}\n")

    task1_end = time.time()
    print(f"Task1 computation time: {task1_end - task1_start}")
