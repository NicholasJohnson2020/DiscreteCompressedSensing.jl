using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")

include("../discreteCompressedSensing.jl")

using Dates, NPZ, Distributions, Random, Images, JSON

function undersample_FT(FT_vec, FT_mat; sample_rate=1)

    dim = size(FT_mat)[1]
    num_indexes = Int(floor(sample_rate * dim))
    indexes = unique(Int.(round.(LinRange(1, dim, num_indexes))))
    undersample_vec = zeros(Complex{Float64}, num_indexes)
    undersample_mat = zeros(Complex{Float64}, (num_indexes, dim))
    for i in 1:num_indexes
        index = indexes[i]
        undersample_vec[i] = FT_vec[index]
        undersample_mat[i, :] = FT_mat[index, :]
    end

    real_vec = vcat(real.(undersample_vec), imag.(undersample_vec))
    real_mat = vcat(real.(undersample_mat), imag.(undersample_mat))

    return real_vec, real_mat, undersample_vec

end

function perturb_FT(flattened_FT; damping_factor=10)

    n = size(flattened_FT)[1]
    real_std = std(real.(flattened_FT))
    imag_std = std(imag.(flattened_FT))
    standard_normal = Normal(0, 1)

    perturbation = flattened_FT
    perturbation += real_std * rand(standard_normal, n) / damping_factor
    perturbation += imag_std * rand(standard_normal, n) * im / damping_factor

    return perturbation

end

METHOD_NAME = ARGS[1]
INPUT_PATH = ARGS[2]
TASK_ID = ARGS[3]

valid_methods = ["Heuristic_Acc",
                 "BPD_Gurobi_Rounding",
                 "SOC_Relax_Rounding",
                 "MISOC",
                 "Cutting_Planes_Warm"]

@assert METHOD_NAME in valid_methods

param_dict = Dict()
open(INPUT_PATH * "params.json", "r") do f
    global param_dict
    dicttxt = JSON.read(f, String)  # file information to string
    param_dict = JSON.parse(dicttxt)  # parse and transform data
    param_dict = JSON.parse(param_dict)
end

DAMPING_FACTOR = param_dict[string(TASK_ID)]["DAMPING_FACTOR"]
SAMPLE_RATE = param_dict[string(TASK_ID)]["SAMPLE_RATE"]
EPSILON_MULTIPLE = param_dict[string(TASK_ID)]["EPSILON_MULTIPLE"]

OUTPUT_PATH = INPUT_PATH * METHOD_NAME * "/"

numerical_threshold = 1e-4

#slice_indexes = collect(30:10:170)
slice_indexes = [60, 100]

FT_mat = npzread(INPUT_PATH * "FT_mat.npy")
basis_mat = npzread(INPUT_PATH * "basis_mat.npy")

experiment_results = Dict()
experiment_results["Method"] = METHOD_NAME
experiment_results["Damping"] = DAMPING_FACTOR
experiment_results["Rate"] = SAMPLE_RATE
experiment_results["Epsilon"] = EPSILON_MULTIPLE
experiment_results["Slices"] = slice_indexes

for slice_index in slice_indexes
    slice_results = Dict()
    slice_results["b_full"] = []
    slice_results["b_observed"] = []
    slice_results["solution"] = []
    slice_results["L2_error"] = []
    slice_results["L1_error"] = []
    slice_results["L0_norm"] = []
    slice_results["ssim"] = []
    slice_results["execution_time"] = []

    if METHOD_NAME in ["BPD_Gurobi_Rounding", "SOC_Relax_Rounding"]
        slice_results["rounded_solution"] = []
        slice_results["rounded_L2_error"] = []
        slice_results["rounded_L1_error"] = []
        slice_results["rounded_L0_norm"] = []
        slice_results["rounded_ssim"] = []
        slice_results["rounded_execution_time"] = []
    end

    if METHOD_NAME == "SOC_Relax_Rounding"
        slice_results["cutting_planes_lb"] = []
    end

    if METHOD_NAME == "Cutting_Planes_Warm"
        slice_results["num_cuts"] = []
    end

    experiment_results[slice_index] = slice_results
end

start_time = now()

for slice_index in slice_indexes

    FT_vec = npzread(INPUT_PATH * "FT_vec_" * string(slice_index) * ".npy")
    image = npzread(INPUT_PATH * "image_" * string(slice_index) * ".npy");

    perturbed_FT = perturb_FT(FT_vec, damping_factor=DAMPING_FACTOR)
    b_observed, FT_mat_observed, _ = undersample_FT(perturbed_FT, FT_mat,
                                                    sample_rate=SAMPLE_RATE)

    A = FT_mat_observed * basis_mat'

    x_full = pinv(A'*A)*A'*b_observed
    full_error = norm(A*x_full-b_observed)^2

    rounding_time = nothing
    n = size(A)[2]
    objective_value = 0
    beta_rounded = zeros(n)
    num_cuts = 0
    if METHOD_NAME == "Heuristic_Acc"
        trial_start = now()
        beta_fitted, _ = exactCompressedSensingHeuristicAcc(A, b_observed,
                                                            EPSILON_MULTIPLE*full_error)
        trial_end_time = now()
    elseif METHOD_NAME == "BPD_Gurobi_Rounding"
        trial_start = now()
        output = basisPursuitDenoising(A, b_observed,
                                       EPSILON_MULTIPLE*full_error,
                                       solver="Gurobi", round_solution=true)
        beta_fitted = output[4]
        beta_rounded = output[2]
        rounding_time = output[5]
        trial_end_time = now()
    elseif METHOD_NAME == "SOC_Relax_Rounding"
        trial_start = now()
        output = perspectiveRelaxation(A, b_observed,
                                       EPSILON_MULTIPLE*full_error,
                                       n, round_solution=true)
        beta_fitted = output[3]
        beta_rounded = output[2]
        objective_value = output[5]
        rounding_time = output[6]
        trial_end_time = now()
    elseif METHOD_NAME == "MISOC"
        trial_start = now()
        beta_fitted, _, _ = perspectiveFormulation(A, b_observed,
                                                   EPSILON_MULTIPLE*full_error,
                                                   n)
        trial_end_time = now()
    elseif METHOD_NAME == "Cutting_Planes_Warm"
        LOAD_PATH = INPUT_PATH * "SOC_Relax_Rounding/"
        warm_start_data = Dict()
        open(LOAD_PATH * "_" * string(TASK_ID) * ".json", "r") do f
            global warm_start_data
            dicttxt = JSON.read(f, String)  # file information to string
            warm_start_data = JSON.parse(dicttxt)  # parse and transform data
            warm_start_data = JSON.parse(warm_start_data)
        end
        upper_bound = warm_start_data[string(slice_index)]["rounded_solution"][1]
        #lower_bound = warm_start_data[string(slice_index)]["cutting_planes_lb"][1]
        lower_bound = 0
        trial_start = now()
        output = CuttingPlanes(A, b_observed, EPSILON_MULTIPLE*full_error, n,
                               lower_bound_obj=lower_Bound, upper_bound_x_sol=upper_bound)
        beta_fitted = output[1]
        num_cuts = output[4]
        trial_end_time = now()
    end

    reconstruction = basis_mat'*beta_fitted
    L2_error = norm(image-reconstruction)^2 / norm(image)^2
    L1_error = norm(image-reconstruction, 1) / norm(image, 1)
    L0_norm = sum(abs.(beta_fitted) .> numerical_threshold)
    n = size(beta_fitted)[1]
    img_width = Int64(n^0.5)
    ssim = assess_ssim(reshape(image, img_width, img_width),
                       reshape(reconstruction, img_width, img_width))
    elapsed_time = Dates.value(trial_end_time - trial_start)

    append!(experiment_results[slice_index]["b_full"], [perturbed_FT])
    append!(experiment_results[slice_index]["b_observed"], [b_observed])
    append!(experiment_results[slice_index]["solution"], [beta_fitted])
    append!(experiment_results[slice_index]["L2_error"], L2_error)
    append!(experiment_results[slice_index]["L1_error"], L1_error)
    append!(experiment_results[slice_index]["L0_norm"], L0_norm)
    append!(experiment_results[slice_index]["ssim"], ssim)
    append!(experiment_results[slice_index]["execution_time"], elapsed_time)

    if METHOD_NAME in ["BPD_Gurobi_Rounding", "SOC_Relax_Rounding"]

        reconstruction = basis_mat'*beta_rounded
        L2_error = norm(image-reconstruction)^2 / norm(image)^2
        L1_error = norm(image-reconstruction, 1) / norm(image, 1)
        L0_norm = sum(abs.(beta_rounded) .> numerical_threshold)
        n = size(beta_rounded)[1]
        img_width = Int64(n^0.5)
        ssim = assess_ssim(reshape(image, img_width, img_width),
                           reshape(reconstruction, img_width, img_width))

        append!(experiment_results[slice_index]["rounded_solution"], [beta_rounded])
        append!(experiment_results[slice_index]["rounded_L2_error"], L2_error)
        append!(experiment_results[slice_index]["rounded_L1_error"], L1_error)
        append!(experiment_results[slice_index]["rounded_L0_norm"], L0_norm)
        append!(experiment_results[slice_index]["rounded_ssim"], ssim)
        append!(experiment_results[slice_index]["rounded_execution_time"], Dates.value(rounding_time))
    end

    if METHOD_NAME == "SOC_Relax_Rounding"
        append!(experiment_results[slice_index]["cutting_planes_lb"],
                objective_value)
    end

    if METHOD_NAME == "Cutting_Planes_Warm"
        append!(experiment_results[slice_index]["num_cuts"], num_cuts)
    end

end

f = open(OUTPUT_PATH * "_" * string(TASK_ID) * ".json","w")
JSON.print(f, JSON.json(experiment_results))
close(f)

total_time = now() - start_time
print("Total execution time: ")
println(total_time)
