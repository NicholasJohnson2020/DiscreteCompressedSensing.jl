using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")

include("../discreteCompressedSensing.jl")

METHOD_NAME = ARGS[1]
INPUT_PATH = ARGS[2]
OUTPUT_PATH_ROOT = ARGS[3]
TASK_ID = ARGS[4]

valid_methods = ["OMP",
                 "BPD_Gurobi_Rounding",
                 "SOC_Relax_Rounding",
                 "MISOC",
                 "MISOC_Backbone",
                 "BnB_Primal",
                 "BnB_Primal_Backbone",
                 "BnB_Primal_Backbone_Rounding",
                 "BnB_Dual",
                 "BnB_Dual_Backbone",
                 "BnB_Dual_Backbone_Rounding"]

@assert METHOD_NAME in valid_methods

param_dict = Dict()
open(INPUT_PATH * "params.json", "r") do f
    global param_dict
    dicttxt = JSON.read(f, String)  # file information to string
    param_dict = JSON.parse(dicttxt)  # parse and transform data
    param_dict = JSON.parse(param_dict)
end

M = param_dict[string(TASK_ID)]["M"]
CR = param_dict[string(TASK_ID)]["CR"]
EPSILON_MULTIPLE = param_dict[string(TASK_ID)]["EPSILON"]

OUTPUT_PATH = OUTPUT_PATH_ROOT * METHOD_NAME * "/"

numerical_threshold = 1e-4
train_size = 30
num_atoms = 2000

#patient_indices = collect((train_size + 1):100)
patient_indices = collect((train_size + 1):35)

sensing_mat_path = INPUT_PATH * "Copmare_ECG_CS-master/BernoulliSample.mat"
sensing_mat = matread(sensing_mat_path)["BernoulliSample"][1:M, :]
dim = size(sensing_mat)[2]

train_data = zeros(dim, train_size)
for index=1:train_size
    ecg_label = "ecg" * string(index)
    ecg_path = INPUT_PATH * "Copmare_ECG_CS-master/data/" * ecg_label * ".mat"
    ecg_signal = matread(ecg_path)[ecg_label]
    train_data[:, index] = ecg_signal
end

atom_dict, _ = ksvd(
    train_data,
    num_atoms,  # the number of atoms in D
    max_iter = 200,  # max iterations of K-SVD
    max_iter_mp = 40,  # max iterations of matching pursuit called in K-SVD
    sparsity_allowance = 0.96
);

A = sensing_mat * atom_dict

experiment_results = Dict()
experiment_results["Method"] = METHOD_NAME
experiment_results["N"] = dim
experiment_results["M"] = M
experiment_results["CR"] = CR
experiment_results["EPSILON"] = EPSILON_MULTIPLE
experiment_results["Indices"] = patient_indices
experiment_results["Train_Size"] = train_size
experiment_results["Num_Atoms"] = num_atoms

for patientID in patient_indices
    patient_results = Dict()
    patient_results["b_full"] = []
    patient_results["b_observed"] = []
    patient_results["solution"] = []
    patient_results["L2_error"] = []
    patient_results["L1_error"] = []
    patient_results["L0_norm"] = []
    patient_results["execution_time"] = []

    if METHOD_NAME == "BPD_Gurobi_Rounding"
        patient_results["rounded_solution"] = []
        patient_results["rounded_L2_error"] = []
        patient_results["rounded_L1_error"] = []
        patient_results["rounded_L0_norm"] = []
        #patient_results["rounded_ssim"] = []
        patient_results["rounded_execution_time"] = []
    end

    if METHOD_NAME == "SOC_Relax_Rounding"
        patient_results["rounded_z_solution"] = []
        patient_results["rounded_z_L2_error"] = []
        patient_results["rounded_z_L1_error"] = []
        patient_results["rounded_z_L0_norm"] = []
        #patient_results["rounded_z_ssim"] = []
        patient_results["rounded_z_execution_time"] = []

        patient_results["rounded_x_solution"] = []
        patient_results["rounded_x_L2_error"] = []
        patient_results["rounded_x_L1_error"] = []
        patient_results["rounded_x_L0_norm"] = []
        #patient_results["rounded_x_ssim"] = []
        patient_results["rounded_x_execution_time"] = []

        #patient_results["cutting_planes_lb"] = []
    end

    if METHOD_NAME in ["BnB_Primal",
                       "BnB_Primal_Backbone",
                       "BnB_Primal_Backbone_Rounding",
                       "BnB_Dual",
                       "BnB_Dual_Backbone",
                       "BnB_Dual_Backbone_Rounding"]

        patient_results["num_nodes"] = []

    end


    experiment_results[patientID] = patient_results
end

start_time = now()

for patientID in patient_indices

    println("Starting Patient " * patientID)

    ecg_label = "ecg" * string(patientID)
    ecg_path = INPUT_PATH * "Copmare_ECG_CS-master/data/" * ecg_label * ".mat"
    if patientID == 75
        ecg_signal = matread(ecg_path)["ecg74"]
    else
        ecg_signal = matread(ecg_path)[ecg_label]
    end

    #b_observed = sensing_mat * ecg_signal
    #epsilon = EPSILON_MULTIPLE * norm(b_observed)^2
    perturbed = ecg_signal + rand(Normal(0, mean(abs.(ecg_signal)) / 4),\
                                  size(ecg_signal)[1])
    b_observed = sensing_mat * perturbed
    epsilon = EPSILON_MULTIPLE * norm(b_observed - sensing_mat * ecg_signal)^2

    rounding_time = nothing
    rounding_time_z = nothing
    rounding_time_x = nothing
    n = size(A)[2]
    gamma = n
    objective_value = 0
    beta_rounded = zeros(n)
    beta_rounded_z = zeros(n)
    beta_rounded_x = zeros(n)
    num_cuts = 0
    num_nodes = 0
    if METHOD_NAME == "OMP"
        trial_start = now()
        beta_fitted, _ = exactCompressedSensingHeuristicAcc(A, b_observed,
                                                            epsilon)
        trial_end_time = now()
    elseif METHOD_NAME == "BPD_Gurobi_Rounding"
        trial_start = now()
        output = basisPursuitDenoising(A, b_observed, epsilon,
                                       norm_function="L2",
                                       solver="Gurobi", round_solution=true)
        beta_fitted = output[4]
        beta_rounded = output[2]
        rounding_time = output[5]
        trial_end_time = now()
    elseif METHOD_NAME == "SOC_Relax_Rounding"
        trial_start = now()
        output = perspectiveRelaxation(A, b_observed,
                                       epsilon,
                                       gamma, round_solution=true)
        beta_fitted = output[5]
        beta_rounded_z = output[2]
        beta_rounded_x = output[4]
        objective_value = output[7]
        rounding_time_x = output[8]
        rounding_time_z = output[9]
        trial_end_time = now()
    elseif METHOD_NAME == "MISOC"
        trial_start = now()
        beta_fitted, _, _ = perspectiveFormulation(A, b_observed, epsilon, gamma,
                                                   norm_function="L2",
                                                   BPD_backbone=false)
        trial_end_time = now()
    elseif METHOD_NAME == "MISOC_Backbone"
        trial_start = now()
        beta_fitted, _, _ = perspectiveFormulation(A, b_observed, epsilon, gamma,
                                                   norm_function="L2",
                                                   BPD_backbone=true)
        trial_end_time = now()
    elseif METHOD_NAME == "BnB_Primal"
        trial_start = now()
        output = CS_BnB(A, b_observed, epsilon, gamma, round_at_nodes=false,
                        norm_function="L2", subproblem_type="primal",
                        BPD_backbone=false)
        beta_fitted = output[1]
        num_nodes = output[4]
        trial_end_time = now()
    elseif METHOD_NAME == "BnB_Primal_Backbone"
        trial_start = now()
        output = CS_BnB(A, b_observed, epsilon, gamma, round_at_nodes=false,
                        norm_function="L2", subproblem_type="primal",
                        BPD_backbone=true)
        beta_fitted = output[1]
        num_nodes = output[4]
        trial_end_time = now()
    elseif METHOD_NAME == "BnB_Primal_Backbone_Rounding"
        trial_start = now()
        output = CS_BnB(A, b_observed, epsilon, gamma, round_at_nodes=true,
                        norm_function="L2", subproblem_type="primal",
                        BPD_backbone=true)
        beta_fitted = output[1]
        num_nodes = output[4]
        trial_end_time = now()
    elseif METHOD_NAME == "BnB_Dual"
        trial_start = now()
        output = CS_BnB(A, b_observed, epsilon, gamma, round_at_nodes=false,
                        norm_function="L2", subproblem_type="dual",
                        BPD_backbone=false)
        beta_fitted = output[1]
        num_nodes = output[4]
        trial_end_time = now()
    elseif METHOD_NAME == "BnB_Dual_Backbone"
        trial_start = now()
        output = CS_BnB(A, b_observed, epsilon, gamma, round_at_nodes=false,
                        norm_function="L2", subproblem_type="dual",
                        BPD_backbone=true)
        beta_fitted = output[1]
        num_nodes = output[4]
        trial_end_time = now()
    elseif METHOD_NAME == "BnB_Dual_Backbone_Rounding"
        trial_start = now()
        output = CS_BnB(A, b_observed, epsilon, gamma, round_at_nodes=true,
                        norm_function="L2", subproblem_type="dual",
                        BPD_backbone=true)
        beta_fitted = output[1]
        num_nodes = output[4]
        trial_end_time = now()
    elseif METHOD_NAME == "Cutting_Planes_Warm"
        LOAD_PATH = OUTPUT_PATH_ROOT * "SOC_Relax_Rounding/"
        warm_start_data = Dict()
        open(LOAD_PATH * "_" * string(TASK_ID) * ".json", "r") do f
            #global warm_start_data
            dicttxt = JSON.read(f, String)  # file information to string
            warm_start_data = JSON.parse(dicttxt)  # parse and transform data
            warm_start_data = JSON.parse(warm_start_data)
        end
        upper_bound = warm_start_data[string(patientID)]["rounded_z_solution"][1]
        lower_bound = warm_start_data[string(patientID)]["cutting_planes_lb"][1]
        trial_start = now()
        output = CuttingPlanes(A, b_observed, epsilon, gamma,
                               lower_bound_obj=lower_bound, upper_bound_x_sol=upper_bound)
        beta_fitted = output[1]
        num_cuts = output[4]
        trial_end_time = now()
    elseif METHOD_NAME == "Exact_Naive_Warm"
        LOAD_PATH = OUTPUT_PATH_ROOT * "Heuristic_Acc/"
        warm_start_data = Dict()
        open(LOAD_PATH * "_" * string(TASK_ID) * ".json", "r") do f
            #global warm_start_data
            dicttxt = JSON.read(f, String)  # file information to string
            warm_start_data = JSON.parse(dicttxt)  # parse and transform data
            warm_start_data = JSON.parse(warm_start_data)
        end
        warm_beta = warm_start_data[string(patientID)]["solution"][1]
        warm_k = warm_start_data[string(patientID)]["L0_norm"][1]
        trial_start = now()
        _, beta_fitted = exactCompressedSensing(A, b_observed,
                                                epsilon,
                                                warm_start_params=(warm_k, warm_beta))
        trial_end_time = now()
    elseif METHOD_NAME == "Exact_Binary_Warm"
        LOAD_PATH = OUTPUT_PATH_ROOT * "Heuristic_Acc/"
        warm_start_data = Dict()
        open(LOAD_PATH * "_" * string(TASK_ID) * ".json", "r") do f
            #global warm_start_data
            dicttxt = JSON.read(f, String)  # file information to string
            warm_start_data = JSON.parse(dicttxt)  # parse and transform data
            warm_start_data = JSON.parse(warm_start_data)
        end
        warm_beta = warm_start_data[string(patientID)]["solution"][1]
        warm_k = warm_start_data[string(patientID)]["L0_norm"][1]
        trial_start = now()
        _, beta_fitted = exactCompressedSensingBinSearch(A, b_observed,
                                                    epsilon,
                                                    warm_start_params=(warm_k, warm_beta))
        trial_end_time = now()
    end

    reconstruction = atom_dict * beta_fitted
    L2_error = norm(ecg_signal-reconstruction)^2 / norm(ecg_signal)^2
    L1_error = norm(ecg_signal-reconstruction, 1) / norm(ecg_signal, 1)
    L0_norm = sum(abs.(beta_fitted) .> numerical_threshold)
    elapsed_time = Dates.value(trial_end_time - trial_start)

    append!(experiment_results[patientID]["b_full"], [ecg_signal])
    append!(experiment_results[patientID]["b_observed"], [b_observed])
    append!(experiment_results[patientID]["solution"], [beta_fitted])
    append!(experiment_results[patientID]["L2_error"], L2_error)
    append!(experiment_results[patientID]["L1_error"], L1_error)
    append!(experiment_results[patientID]["L0_norm"], L0_norm)
    append!(experiment_results[patientID]["execution_time"], elapsed_time)

    if METHOD_NAME == "BPD_Gurobi_Rounding"

        reconstruction = atom_dict * beta_rounded
        L2_error = norm(ecg_signal-reconstruction)^2 / norm(ecg_signal)^2
        L1_error = norm(ecg_signal-reconstruction, 1) / norm(ecg_signal, 1)
        L0_norm = sum(abs.(beta_rounded) .> numerical_threshold)

        append!(experiment_results[patientID]["rounded_solution"], [beta_rounded])
        append!(experiment_results[patientID]["rounded_L2_error"], L2_error)
        append!(experiment_results[patientID]["rounded_L1_error"], L1_error)
        append!(experiment_results[patientID]["rounded_L0_norm"], L0_norm)
        append!(experiment_results[patientID]["rounded_execution_time"], Dates.value(rounding_time))
    end

    if METHOD_NAME == "SOC_Relax_Rounding"
        reconstruction = atom_dict * beta_rounded_z
        L2_error = norm(ecg_signal-reconstruction)^2 / norm(ecg_signal)^2
        L1_error = norm(ecg_signal-reconstruction, 1) / norm(ecg_signal, 1)
        L0_norm = sum(abs.(beta_rounded_z) .> numerical_threshold)

        append!(experiment_results[patientID]["rounded_z_solution"], [beta_rounded_z])
        append!(experiment_results[patientID]["rounded_z_L2_error"], L2_error)
        append!(experiment_results[patientID]["rounded_z_L1_error"], L1_error)
        append!(experiment_results[patientID]["rounded_z_L0_norm"], L0_norm)
        append!(experiment_results[patientID]["rounded_z_execution_time"], Dates.value(rounding_time_z))

        reconstruction = atom_dict * beta_rounded_x
        L2_error = norm(ecg_signal-reconstruction)^2 / norm(ecg_signal)^2
        L1_error = norm(ecg_signal-reconstruction, 1) / norm(ecg_signal, 1)
        L0_norm = sum(abs.(beta_rounded_x) .> numerical_threshold)

        append!(experiment_results[patientID]["rounded_x_solution"], [beta_rounded_x])
        append!(experiment_results[patientID]["rounded_x_L2_error"], L2_error)
        append!(experiment_results[patientID]["rounded_x_L1_error"], L1_error)
        append!(experiment_results[patientID]["rounded_x_L0_norm"], L0_norm)
        append!(experiment_results[patientID]["rounded_x_execution_time"], Dates.value(rounding_time_x))

        #append!(experiment_results[patientID]["cutting_planes_lb"],
        #        objective_value)
    end

    if METHOD_NAME in ["BnB_Primal",
                       "BnB_Primal_Backbone",
                       "BnB_Primal_Backbone_Rounding",
                       "BnB_Dual",
                       "BnB_Dual_Backbone",
                       "BnB_Dual_Backbone_Rounding"]

        append!(experiment_results[patientID]["num_nodes"], num_nodes)
    end

    println("Finished Patient " * patientID)
    println()

end

f = open(OUTPUT_PATH * "_" * string(TASK_ID) * ".json","w")
JSON.print(f, JSON.json(experiment_results))
close(f)

total_time = now() - start_time
print("Total execution time: ")
println(total_time)
