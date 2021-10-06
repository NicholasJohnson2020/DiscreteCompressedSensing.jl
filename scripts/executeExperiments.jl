using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")

include("../discreteCompressedSensing.jl")

using JSON, Dates

function unserialize_matrix(mat)
    n = size(mat[1])[1]
    p = size(mat)[1]
    output = zeros(n, p)
    for i=1:n, j=1:p
        output[i, j] = mat[p][i]
    end
    return output
end;

epsilon_list = collect(1.1:0.1:2)
numerical_threshold = 1e-4

method_name = ARGS[1]
input_path = ARGS[2]
output_path = ARGS[3]
task_ID = ARGS[4]

valid_methods = ["BPD_Gurobi", "BPD_SCS", "Exact_Naive", "Exact_Binary",
                 "MISOC"]

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

        if method_name == "BPD_Gurobi"
            trial_start = now()
            beta_fitted = basisPursuitDenoising(X, Y, epsilon * full_error,
                                                solver="Gurobi")
            trial_end_time = now()
        elseif method_name == "BPD_SCS"
            trial_start = now()
            beta_fitted = basisPursuitDenoising(X, Y, epsilon * full_error,
                                                solver="SCS")
            trial_end_time = now()
        elseif method_name == "Exact_Naive"
            trial_start = now()
            _, beta_fitted = exactCompressedSensing(X, Y, epsilon * full_error)
            trial_end_time = now()
        elseif method_name == "Exact_Binary"
            trial_start = now()
            _, beta_fitted = exactCompressedSensingBinSearch(X, Y,
                                                        epsilon * full_error)
            trial_end_time = now()
        elseif method_name == "Exact_MISOC"
            trial_start = now()
            beta_fitted, _, _ = perspectiveFormulation(X, Y, epsilon*full_error,
                                                       1 / n^2)
            trial_end_time = now()
        end

        residual_error = norm(X * beta_fitted - Y)^2
        beta_error = norm(true_beta - beta_fitted)^2 / norm(true_beta)^2
        fitted_k = sum(abs.(beta_fitted) .> numerical_threshold)
        true_discovery = sum(abs.(beta_fitted[abs.(true_beta) .> 1e-6]) .> numerical_threshold)
        true_discovery /= k_sparse
        discovered_indices = abs.(beta_fitted) .>numerical_threshold
        false_discovery = sum(abs.(true_beta[discovered_indices]) .< 1e-6) / sum(discovered_indices)
        elapsed_time = Dates.value(trial_end_time - trial_start)

        append!(experiment_results[epsilon]["solution"], [beta_fitted])
        append!(experiment_results[epsilon]["residual_error"], residual_error)
        append!(experiment_results[epsilon]["beta_error"], beta_error)
        append!(experiment_results[epsilon]["fitted_k"], fitted_k)
        append!(experiment_results[epsilon]["true_discovery"], true_discovery)
        append!(experiment_results[epsilon]["false_discovery"], false_discovery)
        append!(experiment_results[epsilon]["execution_time"], elapsed_time)

    end

end

f = open(output_path * "_" * string(task_ID) * ".json","w")
JSON.print(f, JSON.json(experiment_results))
close(f)

total_time = now() - start_time
print("Total execution time: ")
println(total_time)
