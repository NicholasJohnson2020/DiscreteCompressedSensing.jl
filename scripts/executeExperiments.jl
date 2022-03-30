using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")

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

epsilon_list = collect(1:0.05:1.5)
numerical_threshold = 1e-4

method_name = ARGS[1]
input_path = ARGS[2]
output_path = ARGS[3]
task_ID = ARGS[4]

valid_methods = ["BPD", "BPD_Rounded", "Exact_Naive", "Exact_Binary",
                 "Exact_Naive_Warm", "Exact_Binary_Warm", "MISOC", "SOC_Relax",
                 "SOC_Relax_Rounded", "Heuristic", "Cutting_Planes",
                 "Cutting_Planes_Warm"]

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
n = param_dict[string(task_ID)]["N"]
p = param_dict[string(task_ID)]["P"]
k = param_dict[string(task_ID)]["K"]
ratio = param_dict[string(task_ID)]["noise_ratio"]

key = Dict{String, Real}(param_dict[string(task_ID)])
experiment_data = synthetic_data[string(key)]

param_dict = nothing
synthetic_data = nothing
GC.gc()

experiment_results = Dict()
experiment_results["Method"] = method_name
experiment_results["N"] = n
experiment_results["P"] = p
experiment_results["K"] = k
experiment_results["noise_to_signal"] = ratio
experiment_results["Trials"] = NUM_TRIALS
experiment_results["epsilon_values"] = epsilon_list

for epsilon in epsilon_list
    epsilon_results = Dict()
    epsilon_results["solution"] = []
    epsilon_results["residual_error"] = []
    epsilon_results["beta_error"] = []
    epsilon_results["fitted_k"] = []
    epsilon_results["true_discovery"] = []
    epsilon_results["false_discovery"] = []
    epsilon_results["execution_time"] = []

    if method_name == "BPD_Rounded"
        epsilon_results["rounded_solution"] = []
        epsilon_results["rounded_residual_error"] = []
        epsilon_results["rounded_beta_error"] = []
        epsilon_results["rounded_fitted_k"] = []
        epsilon_results["rounded_true_discovery"] = []
        epsilon_results["rounded_false_discovery"] = []
        epsilon_results["rounded_execution_time"] = []
    end

    if method_name == "SOC_Relax_Rounded"
        epsilon_results["rounded_x_solution"] = []
        epsilon_results["rounded_x_residual_error"] = []
        epsilon_results["rounded_x_beta_error"] = []
        epsilon_results["rounded_x_fitted_k"] = []
        epsilon_results["rounded_x_true_discovery"] = []
        epsilon_results["rounded_x_false_discovery"] = []
        epsilon_results["rounded_x_execution_time"] = []

        epsilon_results["rounded_z_solution"] = []
        epsilon_results["rounded_z_residual_error"] = []
        epsilon_results["rounded_z_beta_error"] = []
        epsilon_results["rounded_z_fitted_k"] = []
        epsilon_results["rounded_z_true_discovery"] = []
        epsilon_results["rounded_z_false_discovery"] = []
        epsilon_results["rounded_z_execution_time"] = []
    end

    if method_name in ["Cutting_Planes", "Cutting_Planes_Warm"]
        epsilon_results["num_cuts"] = []
    end

    experiment_results[epsilon] = epsilon_results
end

start_time = now()

for trial_num=1:NUM_TRIALS
    X = experiment_data[string(trial_num)]["X"]
    X = unserialize_matrix(X)
    Y = experiment_data[string(trial_num)]["Y"]
    true_k = experiment_data[string(trial_num)]["k"]
    true_beta = experiment_data[string(trial_num)]["beta"]

    beta_full = pinv(X'*X)*X'*Y
    full_error = norm(X*beta_full-Y)^2

    for epsilon in epsilon_list

        rounding_time = nothing
        rounding_time_z = nothing
        rounding_time_x = nothing
        gamma = n^2
        objective_value = 0
        beta_rounded = zeros(n)
        beta_rounded_z = zeros(n)
        beta_rounded_x = zeros(n)
        num_cuts = 0

        if method_name == "BPD"
            trial_start = now()
            _, beta_fitted = basisPursuitDenoising(X, Y, epsilon * full_error,
                                                   solver="Gurobi",
                                                   round_solution=false)
            trial_end_time = now()
        elseif method_name == "BPD_Rounded"
            trial_start = now()
            output = basisPursuitDenoising(X, Y, epsilon * full_error,
                                           solver="Gurobi", round_solution=true)
            beta_fitted = output[4]
            beta_rounded = output[2]
            rounding_time = output[5]
            trial_end_time = now()
        elseif method_name == "Exact_Naive"
            trial_start = now()
            _, beta_fitted = exactCompressedSensing(X, Y, epsilon * full_error,
                                                    warm_start=false)
            trial_end_time = now()
        elseif method_name == "Exact_Binary"
            trial_start = now()
            _, beta_fitted = exactCompressedSensingBinSearch(X, Y,
                                                        epsilon * full_error,
                                                        warm_start=false)
            trial_end_time = now()
        elseif method_name == "Exact_Naive_Warm"
            trial_start = now()
            _, beta_fitted = exactCompressedSensing(X, Y, epsilon * full_error,
                                                    warm_start=true)
            trial_end_time = now()
        elseif method_name == "Exact_Binary_Warm"
            trial_start = now()
            _, beta_fitted = exactCompressedSensingBinSearch(X, Y,
                                                        epsilon * full_error,
                                                        warm_start=true)
            trial_end_time = now()
        elseif method_name == "MISOC"
            trial_start = now()
            beta_fitted, _, _ = perspectiveFormulation(X, Y, epsilon*full_error,
                                                       gamma)
            trial_end_time = now()
        elseif method_name == "SOC_Relax"
            trial_start = now()
            beta_fitted, _, _ = perspectiveRelaxation(X, Y, epsilon*full_error,
                                                      gamma, round_solution=false)
            trial_end_time = now()
        elseif method_name == "SOC_Relax_Rounded"
            trial_start = now()
            output = perspectiveRelaxation(X, Y, epsilon*full_error,
                                           gamma, round_solution=true)
            beta_fitted = output[5]
            beta_rounded_z = output[2]
            beta_rounded_x = output[4]
            objective_value = output[7]
            rounding_time_x = output[8]
            rounding_time_z = output[9]
            trial_end_time = now()
        elseif method_name == "Heuristic"
            trial_start = now()
            beta_fitted, _ = exactCompressedSensingHeuristicAcc(X, Y,
                                                             epsilon*full_error)
            trial_end_time = now()
        elseif method_name == "Cutting_Planes"
            trial_start = now()
            output = CuttingPlanes(X, Y, epsilon * full_error, gamma)
            beta_fitted = output[1]
            num_cuts = output[4]
            trial_end_time = now()
        elseif method_name == "Cutting_Planes_Warm"
            load_path = input_path * "Heuristic/"
            warm_start_data = Dict()
            open(load_path * "_" * string(TASK_ID) * ".json", "r") do f
                #global warm_start_data
                dicttxt = JSON.read(f, String)  # file information to string
                warm_start_data = JSON.parse(dicttxt)  # parse and transform data
                warm_start_data = JSON.parse(warm_start_data)
            end
            upper_bound = warm_start_data[string(slice_index)]["solution"][trial_num]
            trial_start = now()
            output = CuttingPlanes(X, Y, epsilon * full_error, gamma,
                                   upper_bound_x_sol=upper_bound)
            beta_fitted = output[1]
            num_cuts = output[4]
            trial_end_time = now()
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

        append!(experiment_results[epsilon]["solution"], [beta_fitted])
        append!(experiment_results[epsilon]["residual_error"], residual_error)
        append!(experiment_results[epsilon]["beta_error"], beta_error)
        append!(experiment_results[epsilon]["fitted_k"], fitted_k)
        append!(experiment_results[epsilon]["true_discovery"], true_discovery)
        append!(experiment_results[epsilon]["false_discovery"], false_discovery)
        append!(experiment_results[epsilon]["execution_time"], elapsed_time)

        if method_name == "BPD_Rounded"
            residual_error = norm(X * beta_rounded - Y)^2
            beta_error = norm(true_beta - beta_rounded)^2 / norm(true_beta)^2
            fitted_k = sum(abs.(beta_rounded) .> numerical_threshold)
            true_discovery = sum(abs.(beta_rounded[abs.(true_beta) .> 1e-6]) .> numerical_threshold)
            true_discovery /= true_k
            discovered_indices = abs.(beta_rounded) .>numerical_threshold
            if fitted_k == 0
                false_discovery = 0
            else
                false_discovery = sum(abs.(true_beta[discovered_indices]) .< 1e-6) / sum(discovered_indices)
            end
            elapsed_time = Dates.value(rounding_time)

            append!(experiment_results[epsilon]["rounded_solution"],
                    [beta_rounded])
            append!(experiment_results[epsilon]["rounded_residual_error"],
                    residual_error)
            append!(experiment_results[epsilon]["rounded_beta_error"],
                    beta_error)
            append!(experiment_results[epsilon]["rounded_fitted_k"],
                    fitted_k)
            append!(experiment_results[epsilon]["rounded_true_discovery"],
                    true_discovery)
            append!(experiment_results[epsilon]["rounded_false_discovery"],
                    false_discovery)
            append!(experiment_results[epsilon]["rounded_execution_time"],
                    elapsed_time)

        end

        if method_name == "SOC_Relax_Rounded"
            residual_error = norm(X * beta_rounded_x - Y)^2
            beta_error = norm(true_beta - beta_rounded_x)^2 / norm(true_beta)^2
            fitted_k = sum(abs.(beta_rounded_x) .> numerical_threshold)
            true_discovery = sum(abs.(beta_rounded_x[abs.(true_beta) .> 1e-6]) .> numerical_threshold)
            true_discovery /= true_k
            discovered_indices = abs.(beta_rounded_x) .>numerical_threshold
            if fitted_k == 0
                false_discovery = 0
            else
                false_discovery = sum(abs.(true_beta[discovered_indices]) .< 1e-6) / sum(discovered_indices)
            end
            elapsed_time = Dates.value(rounding_time_x)

            append!(experiment_results[epsilon]["rounded_x_solution"],
                    [beta_rounded])
            append!(experiment_results[epsilon]["rounded_x_residual_error"],
                    residual_error)
            append!(experiment_results[epsilon]["rounded_x_beta_error"],
                    beta_error)
            append!(experiment_results[epsilon]["rounded_x_fitted_k"],
                    fitted_k)
            append!(experiment_results[epsilon]["rounded_x_true_discovery"],
                    true_discovery)
            append!(experiment_results[epsilon]["rounded_x_false_discovery"],
                    false_discovery)
            append!(experiment_results[epsilon]["rounded_x_execution_time"],
                    elapsed_time)

            residual_error = norm(X * beta_rounded_z - Y)^2
            beta_error = norm(true_beta - beta_rounded_z)^2 / norm(true_beta)^2
            fitted_k = sum(abs.(beta_rounded_z) .> numerical_threshold)
            true_discovery = sum(abs.(beta_rounded_z[abs.(true_beta) .> 1e-6]) .> numerical_threshold)
            true_discovery /= true_k
            discovered_indices = abs.(beta_rounded_z) .>numerical_threshold
            if fitted_k == 0
                false_discovery = 0
            else
                false_discovery = sum(abs.(true_beta[discovered_indices]) .< 1e-6) / sum(discovered_indices)
            end
            elapsed_time = Dates.value(rounding_time_z)

            append!(experiment_results[epsilon]["rounded_z_solution"],
                    [beta_rounded])
            append!(experiment_results[epsilon]["rounded_z_residual_error"],
                    residual_error)
            append!(experiment_results[epsilon]["rounded_z_beta_error"],
                    beta_error)
            append!(experiment_results[epsilon]["rounded_z_fitted_k"],
                    fitted_k)
            append!(experiment_results[epsilon]["rounded_z_true_discovery"],
                    true_discovery)
            append!(experiment_results[epsilon]["rounded_z_false_discovery"],
                    false_discovery)
            append!(experiment_results[epsilon]["rounded_z_execution_time"],
                    elapsed_time)
        end

        if method_name in ["Cutting_Planes", "Cutting_Planes_Warm"]
            append!(experiment_results[epsilon]["num_cuts"], num_cuts)
        end

    end

end

f = open(output_path * "_" * string(task_ID) * ".json","w")
JSON.print(f, JSON.json(experiment_results))
close(f)

total_time = now() - start_time
print("Total execution time: ")
println(total_time)
