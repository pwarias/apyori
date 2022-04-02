import DPapyori as DPapyori
import apyori 
import matplotlib.pyplot as plt
import random
import os

input_file = 'csv_files/DataSetA.csv' #'og_output.txt'
MAX_TRANS_LENGTH = 100

def load(dir, package):
    with open(dir) as f:
        transactions = list(package.load_transactions(f, delimiter=','))
        return transactions

def truncate_transactions(t, epsilon):
    distributions = [0 for x in range(MAX_TRANS_LENGTH)]
    t_prime = list()
    max = 0
    for transaction in t:
        if len(transaction) > max:
            max = len(transaction)
        distributions[len(transaction)] += 1
    total_transaction = sum(distributions)
    
    # inject noise into the frequency estimatino
    noise_array = DPapyori.G(epsilon, len(distributions))
    for i in range(len(distributions)):
        distributions[i] += noise_array[i]

    # figure out l
    l = MAX_TRANS_LENGTH
    while(sum(distributions[0:l])/total_transaction >= 0.85):
        l -= 1

    for row in t:
        temp = list()
        if len(row) > l:
            for _ in range(l):
                temp.append(random.choice(row))
            t_prime.append(temp)
        else:
            t_prime.append(row)
    #print(l, "vs ", max)
    return t_prime

def recall(private, correct):
    score = 0
    for result in private.keys():
        if result in correct.keys():
            score += 1
    return score / len(correct)

def precision(private, correct):
    score = 0
    for result in private.keys():
        if result in correct.keys():
            score += 1
    return score / len(private)

def f_score(private, correct):
    prec = precision(private, correct)
    rec = recall(private, correct)
    return 2 * ((prec * rec)/ (prec + rec))

def regular_apriori(transacts_prime, min_confidence = 0.0, max_length = 2):
    results = list(apyori.apriori(transacts_prime, min_confidence = min_confidence, max_length = max_length))
    correct = dict()
    for x in results:
        correct[x[0]] = x[1]
    return correct


def main():
    epsilons = [0.05, 0.1, 0.5, 1, 2, 4, 6, 8]
    recalls = list()
    precisions = list()
    f_scores = list()
    for epsilon in epsilons:
        transacts = load(os.path.abspath(input_file), DPapyori)
        transacts_prime = truncate_transactions(transacts, epsilon) # need to play around with epsilon here
        correct = regular_apriori(transacts, min_confidence = 0.0, max_length = 2)
        private = dict(DPapyori.apriori(transacts_prime, min_confidence = 0.0, max_length = 2, epsilon = 0.1))
        recalls.append(recall(private, correct))
        precisions.append(precision(private, correct))
        f_scores.append(f_score(private, correct))
    
    plt.plot(epsilons, precisions, label='Precision')
    plt.plot(epsilons, recalls, label='Recall')
    plt.plot(epsilons, f_scores, label='F_score')
    plt.xlabel('epsilon value')
    plt.ylabel('resulting values')
    plt.title('measures vs. epsilon')
    plt.legend()
    plt.savefig('results.jpg', bbox_inches = 'tight', dpi = 150)


    # format original apyori output to match DPapyori output
    # just contains the itemset and support
    print(f_score(private, correct))

main()