
import pickle



import numpy as np

import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt



def Scenario_1(data, train_size=1, num_prophets=2):

    """

    Question 1.

    2 Prophets 1 Game.

    You may change the input & output parameters of the function as you wish.

    """



    with open('data.pkl', 'rb') as f:

        labels = pickle.load(f)

    y_train, y_test = labels['train_set'], labels['test_set']



    # to the report:

    best_chosen_count = 0  # to sum in how many experiments choosing the best prophet

    total_average_error = 0  # to report the average error of the selected prophets over the experiments.

    total_estimation_error = 0

    not_worse_than_1_count = 0  # to calculate In how many experiments did you choose a prophet that was not 1% worse

    approximation_error = 0

    average_error_list = []



    for i in range(100):

        game_indices = np.random.choice(10000, train_size, False)

        # selecting a prophet based on the ERM algorithm:

        empirical_risks = []

        for prophet in range(num_prophets):

            results = data["train_set"][prophet, game_indices]

            num_false = np.sum(results != y_train[game_indices])

            empirical_risk = num_false / train_size

            empirical_risks.append(empirical_risk)



        empirical_risks = np.array(empirical_risks)

        min_indices = np.where(empirical_risks == np.min(empirical_risks))[0]

        chosen_prophet = np.random.choice(min_indices)



        # compute the chosen_prophet's average error:

        test_results = data['test_set'][chosen_prophet]

        false_test = np.sum(test_results != y_test)

        average_error = false_test / len(test_results)

        average_error_list.append(average_error)

        total_average_error += average_error



        # approximation error

        approximation_error = np.min(data['true_risk'])



        # estimation error

        estimation_error = data['true_risk'][chosen_prophet] - approximation_error

        total_estimation_error += estimation_error



        if estimation_error == 0:

            best_chosen_count += 1

        if estimation_error <= 0.01:

            not_worse_than_1_count += 1



    print("report results:\n")



    # the average error of the selected prophets over the experiments:

    average_error_over = total_average_error / 100

    print("The average error over the experiments:", average_error_over)

    print("The list of all average errors:", average_error_list)



    # In how many experiments did you choose the best prophet?

    print("we choose the best prophet for", best_chosen_count, "out of 100 experiments")



    # the approximation error:

    print("the approximation error:", approximation_error)



    # the mean  estimation errors over the experiments

    estimation_error_over = total_estimation_error / 100

    print("Mean estimation error over experiments:", estimation_error_over)



    # number of times choosing a prophet with true risk <= best true risk + 0.01:

    print("Number of times choosing a prophet with true risk <= best + 0.01:",

          not_worse_than_1_count)

def Scenario_2(data):

    """

    Question 2.

    2 Prophets 10 Games.

    You may change the input & output parameters of the function as you wish.

    """

    Scenario_1(data, 10, 2)





def Scenario_3(data):

    """

    Question 3.

    500 Prophets 10 Games.

    You may change the input & output parameters of the function as you wish.

    """

    Scenario_1(data, 10, data['train_set'].shape[0])





def Scenario_4(data):

    """

    Question 4.

    500 Prophets 1000 Games.

    You may change the input & output parameters of the function as you wish.

    """

    Scenario_1(data, 1000, data['train_set'].shape[0])





def print_error_table(table, k_list, m_list):

    print("Table of mean errors for each k (rows) and m (columns):\n")

    col_width = 38

    corner = "k \\ m"

    header = f"{corner:<8}" + "".join(f"{'m=' + str(m):<{col_width}}" for m in m_list)

    print(header)

    print("-" * len(header))

    for k in k_list:

        row = f"{'k=' + str(k):<8}"

        for m in m_list:

            approx, est, test = table[(k, m)]

            cell = f"approx={approx:.3f}, est={est:.3f}, test={test:.3f}"

            row += f"{cell:<{col_width}}"

        print(row)



def Scenario_5(data):

    """

    Question 5.

    School of Prophets.

    You may change the input & output parameters of the function as you wish.

    """

    with open('data.pkl', 'rb') as f:

        labels = pickle.load(f)

    y_train, y_test = labels['train_set'], labels['test_set']



    num_prophets = data['train_set'].shape[0]

    num_games_total = data['train_set'].shape[1]

    num_games_test = data['test_set'].shape[1]



    k_list = [2, 5, 10, 50]

    m_list = [1, 10, 50, 1000]

    # for printing the table in the report:

    table = {}



    for k in k_list:

        for m in m_list:

            approximation_sum = 0

            estimation_sum = 0

            test_error_sum = 0

            for i in range(100):

                # new set of k prophets and m training games in every experiment

                prophets_indices = np.random.choice(num_prophets, k, False)

                game_indices = np.random.choice(num_games_total, m, False)

                empirical_risks = {}

                for prophet in prophets_indices:

                    prophet_predictions = data['train_set'][prophet, game_indices]

                    error = np.sum(prophet_predictions != y_train[game_indices])

                    empirical_risk = error / m

                    empirical_risks[prophet] = empirical_risk



                min_risk = min(empirical_risks.values())

                min_prophets = [p for p in empirical_risks if empirical_risks[p] == min_risk]

                best_prophet = np.random.choice(min_prophets)



                # approximation & estimation w.r.t. the sampled k prophets

                selected_true_risks = data['true_risk'][prophets_indices]

                approximation_error = np.min(selected_true_risks)

                approximation_sum += approximation_error

                estimation_error = data['true_risk'][best_prophet] - approximation_error

                estimation_sum += estimation_error



                # evaluate on the full test set

                results = data['test_set'][best_prophet]

                false_test = np.sum(results != y_test)

                test_error_sum += false_test / num_games_test



            mean_approximation = approximation_sum / 100

            mean_estimation = estimation_sum / 100

            mean_test_error = test_error_sum / 100



            print(f"Results for k={k}, m={m}:")

            print("The mean test error of the selected prophet:", mean_test_error)

            print("The mean ERM approximation error over the experiments:", mean_approximation)

            print("The mean ERM estimation error over the experiments:", mean_estimation)



            table[(k, m)] = (mean_approximation, mean_estimation, mean_test_error)



    print_error_table(table, k_list, m_list)



def Scenario_6(data):

    """

    Question 6.

    The Bias-Variance Tradeoff.

    You may change the input & output parameters of the function as you wish.

    """

    with open('data.pkl', 'rb') as f:

        labels = pickle.load(f)

    y_train, y_test = labels['train_set'], labels['test_set']



    hypotheses = ['hypothesis1', 'hypothesis2']

    # The test_error_sum[0] is the test_sum of hypothesis1, and [1] is hypothesis2.

    test_error_sum = [0, 0]

    estimation_error_sum = [0, 0]

    estimation_errors_list = [[], []]

    approx_error = [min(data['hypothesis1']['true_risk']), min(data['hypothesis2']['true_risk'])]

    inx = 0

    for i in range(100):

        # choosing 10 games randomly (same train set for both classes):

        num_games = data['hypothesis1']['train_set'].shape[1]

        game_indices = np.random.choice(num_games, 10, replace=False)

        for hyp in hypotheses:

            if hyp == 'hypothesis1':

                inx = 0

            else:

                inx = 1

            train_set = data[hyp]['train_set']

            num_prophets = train_set.shape[0]

            empirical_risks = {}



            # calculate the empirical risk of the prophets:

            for prophet in range(num_prophets):

                predictions = train_set[prophet, game_indices]

                error = np.sum(predictions != y_train[game_indices])

                empirical_risk = error / 10

                empirical_risks[prophet] = empirical_risk



            min_risk = min(empirical_risks.values())

            min_prophets = [p for p in empirical_risks if empirical_risks[p] == min_risk]

            best_prophet = np.random.choice(min_prophets)



            # question 1: compute the error of the chosen prophets using the test set.

            test_set = data[hyp]['test_set']

            test_predictions = test_set[best_prophet]  # array of the predictions

            test_error = np.sum(test_predictions != y_test) / len(test_predictions)

            test_error_sum[inx] += test_error



            # question 2: estimation error

            estimation_error = data[hyp]['true_risk'][best_prophet] - approx_error[inx]

            estimation_error_sum[inx] += estimation_error



            # save for histogram

            estimation_errors_list[inx].append(estimation_error)

    # question 1: Average test errors

    print("\n===== Average Test Errors =====")

    print(f"{'Hypothesis':<15} {'Average Test Error':>20}")

    print("-" * 35)

    print(f"{'Hypothesis 1':<15} {test_error_sum[0] / 100:>20.4f}")

    print(f"{'Hypothesis 2':<15} {test_error_sum[1] / 100:>20.4f}")

    print("=" * 35)

    # question 2: Approximation and average estimation error

    print("\n===== Approximation Error =====")

    print(f"{'Hypothesis':<15} {'Approximation Error':>20}")

    print("-" * 35)

    print(f"{'Hypothesis 1':<15} {min(data['hypothesis1']['true_risk'])}")

    print(f"{'Hypothesis 2':<15} {min(data['hypothesis2']['true_risk'])}")

    print("-" * 35)

    print("\n===== Average Estimation Error =====")

    print(f"{'Hypothesis':<15} {'Average Estimation Error':>20}")

    print("-" * 35)

    print(f"{'Hypothesis 1':<15} {estimation_error_sum[0] / 100:>20.4f}")

    print(f"{'Hypothesis 2':<15} {estimation_error_sum[1] / 100:>20.4f}")

    print("=" * 35)

    min_range = min(min(estimation_errors_list[0]), min(estimation_errors_list[1]))

    max_range = max(max(estimation_errors_list[0]), max(estimation_errors_list[1]))

    bins = 20



    plt.figure(figsize=(8, 6))

    plt.hist(estimation_errors_list[0], bins=bins, range=(min_range, max_range),

             alpha=0.5, label='Hypothesis 1', color='blue')

    plt.hist(estimation_errors_list[1], bins=bins, range=(min_range, max_range),

             alpha=0.5, label='Hypothesis 2', color='red')



    plt.xlabel("Estimation Error")

    plt.ylabel("Frequency")

    plt.title("Histogram of Estimation Errors for Two Hypothesis Classes")

    plt.legend()

    plt.show()



if __name__ == '__main__':

    scenario_1_2 = "scenario_one_and_two_prophets.pkl"

    with open(scenario_1_2, "rb") as f:

        first_two_data = pickle.load(f)



    print(f'Scenario 1 Results:')

    Scenario_1(first_two_data)



    print(f'Scenario 2 Results:')

    Scenario_2(first_two_data)



    scenario_3_4 = "scenario_three_and_four_prophets.pkl"

    with open(scenario_3_4, "rb") as f:

        three_four_data = pickle.load(f)



    print(f'Scenario 3 Results:')

    Scenario_3(three_four_data)



    print(f'Scenario 4 Results:')

    Scenario_4(three_four_data)



    scenario_5 = "scenario_five_prophets.pkl"

    with open(scenario_5, "rb") as f:

        five_prophets_data = pickle.load(f)

    print(f'Scenario 5 Results:')

    Scenario_5(five_prophets_data)



    scenario_6 = "scenario_six_prophets.pkl"

    with open(scenario_6, "rb") as f:

        six_prophets_data = pickle.load(f)



    print(f'Scenario 6 Results:')

    Scenario_6(six_prophets_data)

