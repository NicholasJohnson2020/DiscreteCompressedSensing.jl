include("../discreteCompressedSensing.jl")

using JSON, Dates

function unserialize_matrix(mat)
    n = size(mat[1])[1]
    p = size(mat)[1]
    output = zeros(n, p)
    for i=1:n, j=1:p
        output[i, j] = mat[j][i]
    end
    return output
end;

numerical_threshold = 1e-4

method_name = ARGS[1]
input_path = ARGS[2]
output_path = ARGS[3] * method_name * "/"
task_ID_input = parse(Int64, ARGS[4])
num_tasks_input = parse(Int64, ARGS[5])
epsilon_BnB = 0.1

#old_valid_methods = ["BPD", "BPD_Rounded", "Exact_Naive", "Exact_Binary",
#                 "Exact_Naive_Warm", "Exact_Binary_Warm", "MISOC", "SOC_Relax",
#                 "SOC_Relax_Rounded", "Heuristic", "Cutting_Planes",
#                 "Cutting_Planes_Warm"]

valid_methods = ["BPD", "BPD_Rounded", "IRWL1", "IRWL1_Rounded", "OMP",
                 "MISOC", "SOC_Relax", "SOC_Relax_Rounded", "BnB_Primal",
                 "BnB_Dual", "SOS"]

@assert method_name in valid_methods

synthetic_data = Dict()
open(input_path * "data.json", "r") do f
    global synthetic_data
    dicttxt = JSON.read(f, String)  # file information to string
    synthetic_data = JSON.parse(dicttxt)  # parse and transform data
    synthetic_data = JSON.parse(synthetic_data)
end

param_dict = Dict()
open(input_path * "params.json", "r") do f
    global param_dict
    dicttxt = JSON.read(f, String)  # file information to string
    param_dict = JSON.parse(dicttxt)  # parse and transform data
    param_dict = JSON.parse(param_dict)
end

NUM_TRIALS = synthetic_data["Trials"]

task_ID_list = collect((task_ID_input+1):num_tasks_input:length(param_dict))

start_time_global = now()

for task_ID in task_ID_list

    global synthetic_data
    global param_dict

    n = param_dict[string(task_ID)]["N"]
    m = param_dict[string(task_ID)]["M"]
    k = param_dict[string(task_ID)]["K"]
    ratio = param_dict[string(task_ID)]["signal_ratio"]
    alpha = param_dict[string(task_ID)]["alpha"]

    key = Dict{String, Real}(param_dict[string(task_ID)])
    experiment_data = synthetic_data[string(key)]

    param_dict = nothing
    synthetic_data = nothing
    GC.gc()

    experiment_results = Dict()
    experiment_results["Method"] = method_name
    experiment_results["N"] = n
    experiment_results["M"] = m
    experiment_results["K"] = k
    experiment_results["signal_to_noise"] = ratio
    experiment_results["alpha"] = alpha
    experiment_results["Trials"] = NUM_TRIALS

    experiment_results["solution"] = []
    experiment_results["residual_error"] = []
    experiment_results["beta_error"] = []
    experiment_results["fitted_k"] = []
    experiment_results["true_discovery"] = []
    experiment_results["false_discovery"] = []
    experiment_results["execution_time"] = []

    if method_name in ["SOC_Relax", "SOC_Relax_Rounded", "SOS"]
        experiment_results["lower_bound"] = []
    end

    if method_name in ["BPD_Rounded", "IRWL1_Rounded", "SOC_Relax_Rounded"]
        experiment_results["rounded_solution"] = []
        experiment_results["rounded_residual_error"] = []
        experiment_results["rounded_beta_error"] = []
        experiment_results["rounded_fitted_k"] = []
        experiment_results["rounded_true_discovery"] = []
        experiment_results["rounded_false_discovery"] = []
        experiment_results["rounded_execution_time"] = []
    end

    if method_name in ["BnB_Primal", "BnB_Dual"]
        experiment_results["root_node_gap"] = []
        experiment_results["num_nodes"] = []
        experiment_results["terminal_nodes"] = []
        experiment_results["lb_history"] = []
        experiment_results["ub_history"] = []
        experiment_results["epsilon_BnB"] = []
    end

    start_time = now()

    for trial_num=1:NUM_TRIALS

        println("Starting trial " * string(trial_num) * " of " * string(NUM_TRIALS))

        X = experiment_data[string(trial_num)]["X"]
        X = unserialize_matrix(X)
        Y = experiment_data[string(trial_num)]["Y"]
        true_k = experiment_data[string(trial_num)]["k"]
        true_beta = experiment_data[string(trial_num)]["beta"]

        rounding_time = nothing
        gamma = n^2
        objective_value = 0
        beta_rounded = zeros(n)

        if method_name == "BPD"
            trial_start = now()
            _, beta_fitted = basisPursuitDenoising(X, Y, alpha * norm(Y)^2,
                                                   round_solution=false)
            trial_end_time = now()
        elseif method_name == "BPD_Rounded"
            trial_start = now()
            _, beta_fitted = basisPursuitDenoising(X, Y, alpha * norm(Y)^2,
                                                   round_solution=false)
            trial_end_time = now()
            rounding_start = now()
            beta_rounded, _ = roundSolution(beta_fitted, X, Y, alpha * norm(Y)^2)
            rounding_time = now() - rounding_start
        elseif method_name == "IRWL1"
            trial_start = now()
            _, beta_fitted, _ = iterativeReweightedL1(X, Y, alpha * norm(Y)^2,
                                                      round_solution=false)
            trial_end_time = now()
        elseif method_name == "IRWL1_Rounded"
            trial_start = now()
            _, beta_fitted, _ = iterativeReweightedL1(X, Y, alpha * norm(Y)^2,
                                                      round_solution=false)
            trial_end_time = now()
            rounding_start = now()
            beta_rounded, _ = roundSolution(beta_fitted, X, Y, alpha * norm(Y)^2)
            rounding_time = now() - rounding_start
        elseif method_name == "OMP"
            trial_start = now()
            beta_fitted, _ = OMP(X, Y, alpha * norm(Y)^2)
            trial_end_time = now()
        elseif method_name == "MISOC"
            trial_start = now()
            beta_fitted, _, _ = perspectiveFormulation(X, Y, alpha * norm(Y)^2,
                                                       gamma)
            trial_end_time = now()
        elseif method_name == "SOC_Relax"
            trial_start = now()
            beta_fitted, _, objective_value = perspectiveRelaxation(X, Y,
                                                    alpha * norm(Y)^2,
                                                    gamma, round_solution=false)
            trial_end_time = now()
        elseif method_name == "SOC_Relax_Rounded"
            trial_start = now()
            beta_fitted, opt_z, objective_value = perspectiveRelaxation(X, Y,
                                                    alpha * norm(Y)^2,
                                                    gamma, round_solution=false)
            trial_end_time = now()
            rounding_start = now()
            beta_rounded, _ = roundSolution(opt_z, X, Y, alpha * norm(Y)^2)
            rounding_time = now() - rounding_start
        elseif method_name == "BnB_Primal"
            trial_start = now()
            output = CS_BnB(X, Y, alpha * norm(Y)^2, gamma,
                            termination_threshold=epsilon_BnB,
                            subproblem_type="primal", BPD_backbone=true,
                            use_default_gamma=false, round_at_nodes=true)
            trial_end_time = now()
            beta_fitted = output[1]
            num_nodes = output[4]
            ub_hist = output[5]
            lb_hist = output[6]
            terminal_nodes = output[8]

            root_ub = ub_hist[1]
            root_lb = lb_hist[1]

            append!(experiment_results["num_nodes"], num_nodes)
            append!(experiment_results["terminal_nodes"], terminal_nodes)
            append!(experiment_results["lb_history"], [lb_hist])
            append!(experiment_results["ub_history"], [ub_hist])
            append!(experiment_results["epsilon_BnB"], epsilon_BnB)
            append!(experiment_results["root_node_gap"], (root_ub-root_lb)/root_ub)
        elseif method_name == "BnB_Dual"
            trial_start = now()
            output = CS_BnB(X, Y, alpha * norm(Y)^2, gamma,
                            termination_threshold=epsilon_BnB,
                            subproblem_type="dual", BPD_backbone=true,
                            use_default_gamma=false, round_at_nodes=true)
            trial_end_time = now()
            beta_fitted = output[1]
            num_nodes = output[4]
            ub_hist = output[5]
            lb_hist = output[6]
            terminal_nodes = output[8]

            root_ub = ub_hist[1]
            root_lb = lb_hist[1]

            append!(experiment_results["num_nodes"], num_nodes)
            append!(experiment_results["terminal_nodes"], terminal_nodes)
            append!(experiment_results["lb_history"], [lb_hist])
            append!(experiment_results["ub_history"], [ub_hist])
            append!(experiment_results["epsilon_BnB"], epsilon_BnB)
            append!(experiment_results["root_node_gap"], (root_ub-root_lb)/root_ub)
        elseif method_name == "SOS"
            trial_start = now()
            objective_value = SOSRelaxation(X, Y, alpha * norm(Y)^2, gamma,
                                            solver_output=false,
                                            use_default_lambda=false,
                                            relaxation_degree=1)
            trial_end_time = now()
            beta_fitted = zeros(n)
        end

        residual_error = norm(X * beta_fitted - Y)^2
        beta_error = norm(true_beta - beta_fitted)^2 / norm(true_beta)^2
        fitted_k = sum(abs.(beta_fitted) .> numerical_threshold)
        true_discovery = sum(abs.(beta_fitted[abs.(true_beta) .> 1e-6]) .> numerical_threshold)
        true_discovery /= true_k
        discovered_indices = abs.(beta_fitted) .>numerical_threshold
        if fitted_k == 0
            false_discovery = 0
        else
            false_discovery = sum(abs.(true_beta[discovered_indices]) .< 1e-6) / sum(discovered_indices)
        end
        elapsed_time = Dates.value(trial_end_time - trial_start)

        append!(experiment_results["solution"], [beta_fitted])
        append!(experiment_results["residual_error"], residual_error)
        append!(experiment_results["beta_error"], beta_error)
        append!(experiment_results["fitted_k"], fitted_k)
        append!(experiment_results["true_discovery"], true_discovery)
        append!(experiment_results["false_discovery"], false_discovery)
        append!(experiment_results["execution_time"], elapsed_time)

        if method_name in ["SOC_Relax", "SOC_Relax_Rounded", "SOS"]
            append!(experiment_results["lower_bound"], objective_value)
        end

        if method_name in ["BPD_Rounded", "IRWL1_Rounded", "SOC_Relax_Rounded"]
            rounded_residual_error = norm(X * beta_rounded - Y)^2
            rounded_beta_error = norm(true_beta - beta_rounded)^2 / norm(true_beta)^2
            rounded_fitted_k = sum(abs.(beta_rounded) .> numerical_threshold)
            rounded_true_discovery = sum(abs.(beta_rounded[abs.(true_beta) .> 1e-6]) .> numerical_threshold)
            rounded_true_discovery /= true_k
            rounded_discovered_indices = abs.(beta_rounded) .>numerical_threshold
            if rounded_fitted_k == 0
                rounded_false_discovery = 0
            else
                rounded_false_discovery = sum(abs.(true_beta[rounded_discovered_indices]) .< 1e-6) / sum(rounded_discovered_indices)
            end
            rounded_elapsed_time = Dates.value(rounding_time)

            append!(experiment_results["rounded_solution"], [beta_rounded])
            append!(experiment_results["rounded_residual_error"],
                    rounded_residual_error)
            append!(experiment_results["rounded_beta_error"],
                    rounded_beta_error)
            append!(experiment_results["rounded_fitted_k"], rounded_fitted_k)
            append!(experiment_results["rounded_true_discovery"],
                    rounded_true_discovery)
            append!(experiment_results["rounded_false_discovery"],
                    rounded_false_discovery)
            append!(experiment_results["rounded_execution_time"],
                    rounded_elapsed_time)
        end

        print("Completed trial $trial_num of $NUM_TRIALS total trials.")

    end

    f = open(output_path * "_" * string(task_ID) * ".json","w")
    JSON.print(f, JSON.json(experiment_results))
    close(f)

    total_time = now() - start_time
    print("Total execution time: ")
    println(total_time)
end
