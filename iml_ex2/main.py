import numpy as np
import matplotlib.pyplot as plt
from knn import KNNClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from helpers import read_data_demo , plot_decision_boundaries

K_VALUES = [1, 10, 100, 1000, 3000]
DIST_MATRICS = ['l1', 'l2']
N_ANOMALIES = 50
MAX_DEPTHS = [1, 2, 4, 6, 10, 20, 50, 100]
MAX_LEAF_NODES = [50, 100, 1000]

def knn_decision_boundaries(results, X_test, Y_test):
    """
    choose k_max and k_min according to the L2 test accuracy, and plot the decision boundaries of 3 models:
    (i) L2 with k_max, (ii) L2 with k_min, (iii) L1 with k_max.
    """
    l2_results = [r for r in results if r['distance_metric'] == 'l2']
    k_max = max(l2_results, key=lambda r: r['test_accuracy'])['k']
    k_min = min(l2_results, key=lambda r: r['test_accuracy'])['k']
    print(f"k_max = {k_max}, k_min = {k_min}")

    models_to_plot = [('(i)', 'l2', k_max), ('(ii)', 'l2', k_min), ('(iii)', 'l1', k_max)]
    for name, metric, k in models_to_plot:
        model = next(r['model'] for r in results
                     if r['distance_metric'] == metric and r['k'] == k)
        plot_decision_boundaries(model, X_test, Y_test,
                                 title=f"{name} distance metric = {metric.upper()}, k = {k}")
        
def print_accuracy_table(table):
    """
    Print the test accuracy table: a row per k value and a column per distance metric.
    """
    line = "+" + "-" * 8 + ("+" + "-" * 14) * len(DIST_MATRICS) + "+"

    print("kNN test accuracy")
    print(line)
    print(f"|{'k':^8}" + "".join(f"|{metric.upper():^14}" for metric in DIST_MATRICS) + "|")
    print(line)
    for i, k in enumerate(K_VALUES):
        print(f"|{k:^8}" + "".join(f"|{accuracy:^14.8f}" for accuracy in table[i]) + "|")
    print(line)

def knn_classification(X_train, Y_train, X_test, Y_test):
    """
    Train a kNN model for every combination of k and distance metric,
    and compute its accuracy on the test set.
    """


    results = []
    table = np.zeros((len(K_VALUES), len(DIST_MATRICS)))

    for i, k in enumerate(K_VALUES):
        for j, metric in enumerate(DIST_MATRICS):
            model = KNNClassifier(k=k, distance_metric=metric)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            accuracy = np.mean(y_pred == Y_test)

            results.append({'model': model,
                            'k': k,
                            'distance_metric': metric,
                            'test_accuracy': accuracy})
            table[i, j] = accuracy

    print_accuracy_table(table)
    return results

def find_anomalies(X_train, Y_train, ad_data):
    """
    Find the anomalies in ad_data using kNN distances to the train set:
    the anomaly score of a sample is the sum of the L2 distances to its 5 nearest neighbors,
    and the N_ANOMALIES samples with the highest scores are defined as anomalies.
    """
    model = KNNClassifier(k=5, distance_metric='l2')
    model.fit(X_train, Y_train)
    distances, idx = model.knn_distance(ad_data)
    distances = np.sqrt(distances)  # faiss do not returns squared L2 distances to save time

    anomaly_scores = np.sum(distances, axis=1)
    anomaly_idx = np.argsort(anomaly_scores)[-N_ANOMALIES:] # 50 coordinates with the highest score

    # create vector where all the anomalies are marked with true and the normals with false
    is_anomaly = np.zeros(len(ad_data), dtype=bool)
    is_anomaly[anomaly_idx] = True
    return is_anomaly


def plot_anomalies(X_train, ad_data, is_anomaly):
    """
    Plot the train points in blac, the normal points of ad_data in blue
    and the anomalous points of ad_data in red.
    """
    plt.scatter(X_train[:, 0], X_train[:, 1], c='black', alpha=0.01)
    plt.scatter(ad_data[~is_anomaly, 0], ad_data[~is_anomaly, 1], c='blue', label='normal')
    plt.scatter(ad_data[is_anomaly, 0], ad_data[is_anomaly, 1], c='red', label='anomaly')

    plt.title('Anomaly detection using kNN (k = 5, L2)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

def train_decision_trees(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    """
    train a decision tree for every combination of max_depth and max_leaf_nodes,
    and compute its training, validation and test accuracies.
    """
    results = []
    for max_depth in MAX_DEPTHS:
        for max_leaf_nodes in MAX_LEAF_NODES:
            tree = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
            tree.fit(X_train, Y_train)

            results.append({'model': tree,
                            'max_depth': max_depth,
                            'max_leaf_nodes': max_leaf_nodes,
                            'train_accuracy': np.mean(tree.predict(X_train) == Y_train),
                            'validation_accuracy': np.mean(tree.predict(X_val) == Y_val),
                            'test_accuracy': np.mean(tree.predict(X_test) == Y_test)})
    print_tree_table(results)
    return results

def print_tree_table(results):
    """
    Print the accuracies of all trees: a row per tree, with its max_depth, max_leaf_nodes
    and its training, validation and test accuracies.
    """
    line = "+" + "-" * 11 + "+" + "-" * 16 + ("+" + "-" * 14) * 3 + "+"

    print("Decision tree accuracy")
    print(line)
    print(f"|{'max_depth':^11}|{'max_leaf_nodes':^16}|{'train':^14}|{'validation':^14}|{'test':^14}|")
    print(line)
    for r in results:
        print(f"|{r['max_depth']:^11}|{r['max_leaf_nodes']:^16}"
              f"|{r['train_accuracy']:^14.8f}|{r['validation_accuracy']:^14.8f}|{r['test_accuracy']:^14.8f}|")
    print(line)

def create_tree_plot(best_tree):
    """
    Return the entry (dict) of the tree with the highest validation accuracy.
    """
    print(f"Best validation tree: max_depth = {best_tree['max_depth']}, "
          f"max_leaf_nodes = {best_tree['max_leaf_nodes']}, "
          f"validation accuracy = {best_tree['validation_accuracy']:.8f}, "
          f"test accuracy = {best_tree['test_accuracy']:.8f}")
    plot_decision_boundaries(best_tree['model'], X_test, Y_test,
                             title=f"Decision tree: max_depth = {best_tree['max_depth']}, "
                                   f"max_leaf_nodes = {best_tree['max_leaf_nodes']}")

def best_validation_tree(results):
    """
    Return the entry (dict) of the tree with the highest validation accuracy.
    """
    return max(results, key=lambda r: r['validation_accuracy'])

def train_forest(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    forest = RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=4)
    forest.fit(X_train, Y_train)
    print(f"random forest: train accuracy = {np.mean(forest.predict(X_train) == Y_train):.8f}, "
          f"validation accuracy = {np.mean(forest.predict(X_val) == Y_val):.8f}, "
          f"test accuracy = {np.mean(forest.predict(X_test) == Y_test):.8f}")
    plot_decision_boundaries(forest, X_test, Y_test,
                             title="Random forest: 300 trees, max_depth = 6")

def train_xgboost_model(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4)
    xgb.fit(X_train, Y_train)
    print(f"XGBoost: train accuracy = {np.mean(xgb.predict(X_train) == Y_train):.8f}, "
          f"validation accuracy = {np.mean(xgb.predict(X_val) == Y_val):.8f}, "
          f"test accuracy = {np.mean(xgb.predict(X_test) == Y_test):.8f}")
    plot_decision_boundaries(xgb, X_test, Y_test,
                             title="XGBoost: 300 trees, max_depth = 6, learning_rate = 0.1")

if __name__ == '__main__':
    np.random.seed(0)

    # Read the data: the first two columns are the coordinates, the third is the state
    train_data, col_names = read_data_demo('train.csv')
    test_data, col_names = read_data_demo('test.csv')
    val_data, col_names = read_data_demo('validation.csv')

    X_train, Y_train = train_data[:, :2], train_data[:, 2].astype(int)
    X_test, Y_test = test_data[:, :2], test_data[:, 2].astype(int)
    X_val, Y_val = val_data[:, :2], val_data[:, 2].astype(int)

    # kNN classification
    knn_results = knn_classification(X_train, Y_train, X_test, Y_test)
    knn_decision_boundaries(knn_results, X_test, Y_test)

    ad_data, ad_col_names = read_data_demo('AD_test.csv')
    is_anomaly = find_anomalies(X_train, Y_train, ad_data)
    plot_anomalies(X_train, ad_data, is_anomaly)

    # decision trees
    tree_results = train_decision_trees(X_train, Y_train, X_val, Y_val, X_test, Y_test)

    best_tree = best_validation_tree(tree_results)
    create_tree_plot(best_tree)

    best_50_leaves = best_validation_tree([r for r in tree_results if r['max_leaf_nodes'] == 50]) 
    create_tree_plot(best_50_leaves)

    best_depth_6 = best_validation_tree([r for r in tree_results if r['max_depth'] <= 6]) 
    create_tree_plot(best_depth_6)

    # random forest with 300 trees of max_depth 6
    train_forest(X_train, Y_train, X_val, Y_val, X_test, Y_test)

    #xgboost model
    train_xgboost_model(X_train, Y_train, X_val, Y_val, X_test, Y_test)

    

    
