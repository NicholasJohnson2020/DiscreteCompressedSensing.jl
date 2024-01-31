include("../discreteCompressedSensing.jl")

method_name = string(ARGS[1])
input_path = ARGS[2]
output_path = input_path * method_name * "/"
task_ID_input = parse(Int64, ARGS[3])
num_tasks_input = parse(Int64, ARGS[4])
epsilon_BnB = 0.1

epsilon_mult = 0.1

gamma_flag = "SQUARE"
gamma_mult = 0.5

valid_methods = ["BPD_Rounded", "IRWL1_Rounded", "OMP", "BnB_Primal",
                 "BnB_Dual"]

@assert method_name in valid_methods

# Load the experiment parameters
#param_dict = Dict()
#open(input_path * "params.json", "r") do f
#    global param_dict
#    dicttxt = JSON.read(f, String)  # file information to string
#    param_dict = JSON.parse(dicttxt)  # parse and transform data
#    param_dict = JSON.parse(param_dict)
#end

# Load the experiment data
A = npzread(input_path * "A.npy")
Y_train_mean = npzread(input_path * "Y_train_mean.npy")
Y_test_pred = npzread(input_path * "Y_test_pred.npy")
Y_train_pred = npzread(input_path * "Y_train_pred.npy")
Y_test = npzread(input_path * "Y_test.npy")
Y_train = npzread(input_path * "Y_train.npy")

numerical_threshold = 1e-4

task_ID_list = collect((task_ID_input+1):num_tasks_input:size(Y_test_pred)[1])
#param_combinations = [(i, j) for i=1:size(Y_test_pred)[1], j=1:length(param_dict)]
#task_ID_list = param_combinations[(task_ID_input+1):num_tasks_input:size(param_combinations)[1]]

# Main loop to execute experiments

start_time = now()

#for param_list in task_ID_list

for TASK_ID in task_ID_list

    #TASK_ID = param_list[1]
    #PARAM_ID = param_list[2]
    #LABEL_ID = (PARAM_ID - 1) * length(param_dict) + TASK_ID

    println("Starting Example " * string(TASK_ID))

    #gamma_mult = param_dict[string(PARAM_ID)]["GAMMA_MULT"]
    #gamma_flag = param_dict[string(PARAM_ID)]["GAMMA_FLAG"]

    # Create dictionary to store experiment results
    experiment_results = Dict()
    experiment_results["METHOD"] = method_name
    experiment_results["TASK_ID"] = TASK_ID
    experiment_results["EPSILON"] = epsilon_mult
    experiment_results["INPUT_PATH"] = input_path
    experiment_results["GAMMA_MULT"] = gamma_mult
    experiment_results["GAMMA_FLAG"] = gamma_flag

    b_observed = Y_test_pred[TASK_ID, :]
    epsilon = epsilon_mult * norm(b_observed)^2

    rounding_time = nothing
    n = size(A)[2]
    gamma = gamma_mult
    if gamma_flag == "SQUARE_ROOT"
        gamma = gamma * sqrt(n)
    elseif gamma_flag == "LINEAR"
        gamma = gamma * n
    elseif gamma_flag == "SQUARE"
        gamma = gamma * n^2
    end

    beta_rounded = zeros(n)
    num_cuts = 0
    num_nodes = 0

    # Switch to execute the specified method
    if method_name == "OMP"
        trial_start = now()
        beta_fitted, _ = OMP(A, b_observed, epsilon)
        trial_end_time = now()
    elseif method_name == "BPD_Rounded"
        trial_start = now()
        _, beta_fitted = basisPursuitDenoising(A, b_observed, epsilon,
                                               round_solution=false)
        trial_end_time = now()
        rounding_start = now()
        beta_rounded, _ = roundSolution(beta_fitted, A, b_observed, epsilon)
        rounding_time = now() - rounding_start
    elseif method_name == "IRWL1_Rounded"
        trial_start = now()
        _, beta_fitted, _ = iterativeReweightedL1(A, b_observed, epsilon,
                                                  round_solution=false)
        trial_end_time = now()
        rounding_start = now()
        beta_rounded, _ = roundSolution(beta_fitted, A, b_observed, epsilon)
        rounding_time = now() - rounding_start
    elseif method_name == "BnB_Primal"
        trial_start = now()
        output = CS_BnB(A, vec(b_observed), epsilon, gamma,
                        termination_threshold=epsilon_BnB,
                        subproblem_type="primal", BPD_backbone=true,
                        use_default_gamma=false, round_at_nodes=true,
                        cutoff_time=2)
        beta_fitted = output[1]
        num_nodes = output[4]
        trial_end_time = now()
    elseif method_name == "BnB_Dual"
        trial_start = now()
        output = CS_BnB(A, vec(b_observed), epsilon, gamma,
                        termination_threshold=epsilon_BnB,
                        subproblem_type="dual", BPD_backbone=true,
                        use_default_gamma=false, round_at_nodes=true,
                        cutoff_time=2)
        beta_fitted = output[1]
        num_nodes = output[4]
        trial_end_time = now()
    end

    # Compute the performance measures of the returned solution
    L2_error = norm(beta_fitted - Y_test[TASK_ID, :]) ^ 2
    L2_error_rel = L2_error / norm(Y_test[TASK_ID, :]) ^ 2
    L0_norm = sum(abs.(beta_fitted) .> numerical_threshold)

    true_ind = findall(==(1), Y_test[TASK_ID, :])
    true_indC = findall(==(0), Y_test[TASK_ID, :])
    selected_ind = findall(==(1), abs.(beta_fitted) .> numerical_threshold)
    selected_indC = findall(==(0), abs.(beta_fitted) .> numerical_threshold)
    TPR = size(findall(in(true_ind), selected_ind))[1] / size(true_ind)[1]
    TNR = size(findall(in(true_indC), selected_indC))[1] / size(true_indC)[1]
    acc = sum((abs.(beta_fitted) .> 1e-4) .== (abs.(Y_test[TASK_ID, :]) .> 1e-4))
    acc /= size(beta_fitted)[1]

    elapsed_time = Dates.value(trial_end_time - trial_start)

    # Store the performance measures of the returned solution
    experiment_results["b_full"] = Y_test[TASK_ID, :]
    experiment_results["b_observed"] = b_observed
    experiment_results["solution"] = beta_fitted
    experiment_results["L2_error"] = L2_error
    experiment_results["L2_error_rel"] = L2_error_rel
    experiment_results["L0_norm"] = L0_norm
    experiment_results["TPR"] = TPR
    experiment_results["TNR"] = TNR
    experiment_results["accuracy"] = acc
    experiment_results["execution_time"] = elapsed_time

    # Compute and store performance measures of rounded solutions
    if method_name in ["BPD_Rounded", "IRWL1_Rounded", "SOC_Relax_Rounded"]

        L2_error = norm(beta_rounded - Y_test[TASK_ID, :]) ^ 2
        L2_error_rel = L2_error / norm(Y_test[TASK_ID, :]) ^ 2
        L0_norm = sum(abs.(beta_rounded) .> numerical_threshold)

        selected_ind = findall(==(1), abs.(beta_rounded) .> numerical_threshold)
        selected_indC = findall(==(0), abs.(beta_rounded) .> numerical_threshold)
        TPR = size(findall(in(true_ind), selected_ind))[1] / size(true_ind)[1]
        TNR = size(findall(in(true_indC), selected_indC))[1] / size(true_indC)[1]
        acc = sum((abs.(beta_rounded) .> 1e-4) .== (abs.(Y_test[TASK_ID, :]) .> 1e-4))
        acc /= size(beta_rounded)[1]

        experiment_results["rounded_solution"] = beta_rounded
        experiment_results["rounded_L2_error"] = L2_error
        experiment_results["rounded_L2_error_rel"] = L2_error_rel
        experiment_results["rounded_L0_norm"] = L0_norm
        experiment_results["rounded_TPR"] = TPR
        experiment_results["rounded_TNR"] = TNR
        experiment_results["rounded_accuracy"] = acc
        experiment_results["rounded_execution_time"] = Dates.value(rounding_time)
    end

    if method_name in ["BnB_Primal", "BnB_Dual"]
        experiment_results["num_nodes"] = num_nodes
    end

    println("Finishing Example " * string(TASK_ID))
    println()

    # Save the results to file
    f = open(output_path * "_" * string(TASK_ID) * ".json","w")
    JSON.print(f, JSON.json(experiment_results))
    close(f)

end

total_time = now() - start_time
print("Total execution time: ")
println(total_time)
