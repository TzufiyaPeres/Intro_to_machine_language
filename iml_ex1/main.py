import pickle
import numpy as np
import matplotlib.pyplot as plt

K_LIST = [2, 5, 10, 50]
M_LIST = [1, 10, 50, 1000]


def run_erm_experiment(num_games_per_experiment, experiments_num):
    """
    Helper function to run ERM experiments with a given number of games per experiment
    and a given number of total experiments.
    """
    with open('scenario_one_and_two_prophets.pkl', 'rb') as f:
        data = pickle.load(f)

    # The true outcomes of the games
    with open('data.pkl', 'rb') as f:
        games_outcomes = pickle.load(f)
    res_train, res_test = games_outcomes['train_set'], games_outcomes['test_set']

    train_prophets = data.get('train_set') 
    true_risks = data.get('true_risk')
    test_prophets = data.get('test_set')

    #print(f"scenario_one_and_two_prophets.pkl contain:\n {data}")
    
    total_games = train_prophets.shape[1]

    best_was_chosen_count = 0
    selected_prophet_risks = []
    test_errors_of_selected = []
    estimation_errors = []

    bias_inx = np.argmin(true_risks)
    approximation_error = true_risks[bias_inx]

    for _ in range(experiments_num):
        # Sample multiple game indices randomly based on the requested number of games
        random_game_indices = np.random.choice(total_games, size=num_games_per_experiment, replace=False)

        # Get the ERM of each prophet averaged over the sampled games
        empirical_risks = np.mean(train_prophets[:, random_game_indices] != res_train[random_game_indices], axis=1)
        min_risk = np.min(empirical_risks)

        #print(f"empirical risks: {empirical_risks}")

        # In case of 2+ prophets choose randomly
        best_prophets_indices = np.where(empirical_risks == min_risk)[0]
        chosen_prophet_idx = np.random.choice(best_prophets_indices)
        
        # Get the true risk of the prophet selected by ERM
        chosen_true_risk = true_risks[chosen_prophet_idx]
        selected_prophet_risks.append(chosen_true_risk)

        # Check the chosen prophet on the test set (fraction of wrong predictions)
        chosen_test_error = np.mean(test_prophets[chosen_prophet_idx] != res_test)
        test_errors_of_selected.append(chosen_test_error)
 
        current_estimation_error = chosen_true_risk - approximation_error
        estimation_errors.append(current_estimation_error)
        if current_estimation_error == 0:
            best_was_chosen_count += 1
 
    mean_selected_risk = np.mean(selected_prophet_risks)
    mean_test_error = np.mean(test_errors_of_selected)
    mean_estimation_error = np.mean(estimation_errors)
 
    print(f"Average Test Error of Selected Prophets: {mean_test_error}")
    print(f"Average True Risk of Selected Prophets: {mean_selected_risk}")
    print(f"Times Best Prophet Chosen: {best_was_chosen_count} out of {experiments_num}")
    print(f"Approximation Error: {approximation_error}")
    print(f"Mean Estimation Error: {mean_estimation_error}")

def run_scenario_unified_risks(num_games_per_experiment, experiments_num):
    """
    Helper function to run ERM experiments for Scenario 3:
    Many prophets, training set of 10 games, repeated for 100 experiments.
    """
    with open('scenario_three_and_four_prophets.pkl', 'rb') as f:
        data = pickle.load(f)

    # The true outcomes of the games
    with open('data.pkl', 'rb') as f:
        games_outcomes = pickle.load(f)
    res_train, res_test = games_outcomes['train_set'], games_outcomes['test_set']
    
    train_prophets = data.get('train_set')
    test_prophets = data.get('test_set')  
    true_risks = data.get('true_risk') 
    #print(f"scenario_three_and_four_prophets.pkl contain:\n {data}")
    
    total_train_games = train_prophets.shape[1]

    best_was_chosen_count = 0
    within_threshold_count = 0
    test_errors_of_selected = []
    estimation_errors = []

    # Find the absolute best prophet in the hypothesis class
    bias_inx = np.argmin(true_risks)
    approximation_error = true_risks[bias_inx]
    #print(f"approximation err: {approximation_error}")

    for _ in range(experiments_num):
        # Sample unique games randomly without replacement
        random_game_indices = np.random.choice(total_train_games, size=num_games_per_experiment, replace=False)

        # Calculate empirical risk for each prophet over the 10 sampled games
        empirical_risks = np.mean(train_prophets[:, random_game_indices] != res_train[random_game_indices], axis=1)
        min_risk = np.min(empirical_risks)

        bias_inx = np.where(empirical_risks == min_risk)[0]
        chosen_prophet_idx = np.random.choice(bias_inx)      

        chosen_true_risk = true_risks[chosen_prophet_idx]

        if chosen_true_risk <= approximation_error + 0.01:
            within_threshold_count += 1

        #check the chosen prophet on the test set
        chosen_test_error = np.mean(test_prophets[chosen_prophet_idx] != res_test)
        test_errors_of_selected.append(chosen_test_error)

        current_estimation_error = chosen_true_risk - approximation_error

        if current_estimation_error == 0:
            best_was_chosen_count += 1
        
        estimation_errors.append(current_estimation_error)
    

    mean_selected_risk = np.mean(test_errors_of_selected)
    mean_estimation_error = np.mean(estimation_errors)

    print(f"Average Error of Selected Prophets: {mean_selected_risk}")
    print(f"Times Best Prophet Chosen: {best_was_chosen_count} out of {experiments_num}")
    print(f"Times Chosen Prophet was within R(h*) + 0.01: {within_threshold_count} out of {experiments_num}")
    print(f"Approximation Error: {approximation_error}")
    print(f"Mean Estimation Error: {mean_estimation_error}")


def run_school_experiment(k, m, data, res_train, res_test, experiments_num=100):
    """
    Helper function for Scenario 5.
    """
    train_prophets = data.get('train_set')
    test_prophets = data.get('test_set')
    true_risks = data.get('true_risk')
 
    num_prophets, total_games = train_prophets.shape
 
    test_errors = []
    approximation_errors = []
    estimation_errors = []
 
    for _ in range(experiments_num):
        # Sample a new set of k prophets and m training games
        prophets_indices = np.random.choice(num_prophets, size=k, replace=False)
        games_indices = np.random.choice(total_games, size=m, replace=False)
 
        # Empirical risk of each of the k sampled prophets over the m sampled games
        sampled_predictions = train_prophets[np.ix_(prophets_indices, games_indices)]
        empirical_risks = np.mean(sampled_predictions != res_train[games_indices], axis=1)
        min_risk = np.min(empirical_risks)
 
        # In case of 2+ prophets choose randomly
        best_local_indices = np.where(empirical_risks == min_risk)[0]
        chosen_local_idx = np.random.choice(best_local_indices)
        chosen_prophet_idx = prophets_indices[chosen_local_idx]
 
        # Approximation error: true risk of the best prophet among the k sampled ones
        approximation_error = np.min(true_risks[prophets_indices])
        approximation_errors.append(approximation_error)
 
        # Estimation error: true risk of the chosen prophet minus the approximation error
        estimation_errors.append(true_risks[chosen_prophet_idx] - approximation_error)
 
        # Evaluate the chosen prophet on the full test set
        test_errors.append(np.mean(test_prophets[chosen_prophet_idx] != res_test))
 
    return (("Test Error", np.mean(test_errors)),
            ("Approximation Error", np.mean(approximation_errors)),
            ("Estimation Error", np.mean(estimation_errors)))

def print_school_table(table, k_list, m_list):
    """
    Prints the results table
    """
    col_width = 40
    header = f"{'k \\ m':<8}| " + "".join(f"{'m=' + str(m):<{col_width}}| " for m in m_list)
    print(header)
    print("-" * len(header))
    for k in k_list:
        # one line per metric, so each cell is clearly labeled
        for metric_idx in range(3):
            row = f"{'k=' + str(k) if metric_idx == 0 else '':<8}| "
            for m in m_list:
                label, value = table[(k, m)][metric_idx]
                row += f"{label + ': ' + format(value, '.4f'):<{col_width}}| "
            print(row)
        print("-" * len(header))

def Scenario_1():
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """
    run_erm_experiment(num_games_per_experiment=1, experiments_num=100)

def Scenario_2():
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    run_erm_experiment(num_games_per_experiment=10, experiments_num=100)


def Scenario_3():
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    run_scenario_unified_risks(num_games_per_experiment=10,experiments_num=100)


def Scenario_4():
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    run_scenario_unified_risks(num_games_per_experiment=1000, experiments_num=100)


def Scenario_5():
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """
    with open('scenario_five_prophets.pkl', 'rb') as f:
            data = pickle.load(f)
 
    # The true outcomes of the games
    with open('data.pkl', 'rb') as f:
        games_outcomes = pickle.load(f)
    res_train, res_test = games_outcomes['train_set'], games_outcomes['test_set']
 

 
    table = {}
    for k in K_LIST:
        for m in M_LIST:
            table[(k, m)] = run_school_experiment(k, m, data, res_train, res_test, experiments_num=100)

    #print(table)
    print_school_table(table, K_LIST, M_LIST)

def Scenario_6():
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    with open('scenario_six_prophets.pkl', 'rb') as f:
        data = pickle.load(f)
 
    # The true outcomes of the games
    with open('data.pkl', 'rb') as f:
        games_outcomes = pickle.load(f)
    res_train, res_test = games_outcomes['train_set'], games_outcomes['test_set']
 
    hypotheses_names = ['hypothesis1', 'hypothesis2']
    experiments_num = 100
    num_games_per_experiment = 10
    total_games = data['hypothesis1']['train_set'].shape[1]
 
    test_errors = {name: [] for name in hypotheses_names}
    estimation_errors = {name: [] for name in hypotheses_names}
    approximation_errors = {name: np.min(data[name]['true_risk']) for name in hypotheses_names}
 
    for _ in range(100):
        # The same train set is used for both classes, so only the class itself makes the difference
        random_game_indices = np.random.choice(total_games, size=10, replace=False)
 
        for name in hypotheses_names:
            train_prophets = data[name]['train_set']
            test_prophets = data[name]['test_set']
            true_risks = data[name]['true_risk']
 
            # empirical risk of each prophet over the sampled games
            empirical_risks = np.mean(train_prophets[:, random_game_indices] != res_train[random_game_indices], axis=1)
            min_risk = np.min(empirical_risks)
 
            # In case of 2+ prophets choose randomly
            best_prophets_indices = np.where(empirical_risks == min_risk)[0]
            chosen_prophet_idx = np.random.choice(best_prophets_indices)
 
            # Error of the chosen prophet on the test set
            test_errors[name].append(np.mean(test_prophets[chosen_prophet_idx] != res_test))

            estimation_errors[name].append(true_risks[chosen_prophet_idx] - approximation_errors[name])
 
    for name in hypotheses_names:
        print(f"{name}:")
        print(f"  Average Test Error: {np.mean(test_errors[name])}")
        print(f"  Approximation Error: {approximation_errors[name]}")
        print(f"  Mean Estimation Error: {np.mean(estimation_errors[name])}")
 
    # Histogram of the estimation errors - same range and same number of bins for both classes
    all_estimation_errors = estimation_errors['hypothesis1'] + estimation_errors['hypothesis2']
    hist_range = (min(all_estimation_errors), max(all_estimation_errors))
    num_bins = 20
 
    plt.figure(figsize=(8, 6))
    plt.hist(estimation_errors['hypothesis1'], bins=num_bins, range=hist_range,
             alpha=0.5, label='Hypothesis 1 (5 prophets)', color='blue')
    plt.hist(estimation_errors['hypothesis2'], bins=num_bins, range=hist_range,
             alpha=0.5, label='Hypothesis 2 (500 prophets)', color='red')
    plt.xlabel("Estimation Error")
    plt.ylabel("Frequency")
    plt.title("Histogram of Estimation Errors (100 experiments)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    

    print(f'Scenario 1 Results:')
    Scenario_1()

    print(f'Scenario 2 Results:')
    Scenario_2()

    print(f'Scenario 3 Results:')
    Scenario_3()

    print(f'Scenario 4 Results:')
    Scenario_4()

    print(f'Scenario 5 Results:')
    Scenario_5()

    print(f'Scenario 6 Results:')
    Scenario_6()
