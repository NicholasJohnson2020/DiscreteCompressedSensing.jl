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

valid_methods = ["BPD_Gurobi", "Exact_Naive_Warm", "Heuristic"]

@assert METHOD_NAME in valid_methods

param_dict = Dict()
open(INPUT_PATH * "params.json", "r") do f
    global param_dict
    dicttxt = JSON.read(f, String)  # file information to string
    param_dict = JSON.parse(dicttxt)  # parse and transform data
    param_dict = JSON.parse(param_dict)
end

DAMPING_FACTOR = param_dict[string(task_ID)]["DAMPING_FACTOR"]
SAMPLE_RATE = param_dict[string(task_ID)]["SAMPLE_RATE"]
EPSILON_MULTIPLE = param_dict[string(task_ID)]["EPSILON_MULTIPLE"]

OUTPUT_PATH = INPUT_PATH * METHOD_NAME * "/"

numerical_threshold = 1e-4

slice_indexes = collect(30:10:170)

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

    if method_name == "BPD_Gurobi"
        trial_start = now()
        _, beta_fitted = basisPursuitDenoising(A, b_observed,
                                               EPSILON_MULTIPLE*full_error,
                                               solver="Gurobi")
        trial_end_time = now()
    elseif method_name == "Exact_Binary_Warm"
        trial_start = now()
        _, beta_fitted = exactCompressedSensingBinSearch(A, b_observed,
                                                         EPSILON_MULTIPLE*full_error,
                                                         warm_start=true)
        trial_end_time = now()
    elseif METHOD_NAME == "Heuristic"
        trial_start = now()
        beta_fitted, _ = exactCompressedSensingHeuristic(A, b_observed,
                                                         EPSILON_MULTIPLE*full_error)
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

end

f = open(OUTPUT_PATH * "_" * string(TASK_ID) * ".json","w")
JSON.print(f, JSON.json(experiment_results))
close(f)

total_time = now() - start_time
print("Total execution time: ")
println(total_time)
