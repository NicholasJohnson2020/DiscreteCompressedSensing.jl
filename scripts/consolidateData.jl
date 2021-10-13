using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")
using JSON, LinearAlgebra, Statistics, DataFrames, CSV

METHOD_NAME = ARGS[1]
NUM_CONFIGS = parse(Int64, ARGS[2])
INPUT_PATH = ARGS[3]
OUTPUT_PATH = INPUT_PATH * METHOD_NAME * "_aggrData.csv"

eval_rank_sparse = METHOD_NAME in ["PCP_Mosek",
                                   "PCP_SCS",
                                   "Projected_Mosek",
                                   "Projected_SCS"]

report_lambda_mu = METHOD_NAME in ["Our_Exact",
                                   "Our_Fast",
                                   "Our_Exact_Bound",
                                   "Our_Fast_Bound"]

RANK_THRESHOLD = 1e-2
SPARSE_THRESHOLD = 1e-2

function unserialize_matrix(mat)
    n = size(mat)[1]
    output = zeros(n, n)
    for i=1:n, j=1:n
        output[i, j] = mat[i][j]
    end
    return output
end

function compute_rank_sparse(solution; rank_threshold=RANK_THRESHOLD,
   sparse_threshold=SPARSE_THRESHOLD)

   X = unserialize_matrix(solution[1])
   Y = unserialize_matrix(solution[2])

   evals_X = svd(X).S
   rank_X = sum(abs.(evals_X) .> rank_threshold)
   sparse_Y = sum(abs.(Y) .> sparse_threshold)

   return rank_X, sparse_Y

end

df = DataFrame(N=Int64[], k_rank=Int64[], k_sparse=Int64[],
               signal_to_noise=Int64[], D_error=Float64[], D_error_std=Float64[],
               L_error=Float64[], L_error_std=Float64[], S_error=Float64[],
               S_error_std=Float64[], true_disc=Float64[], true_disc_std=Float64[],
               false_disc=Float64[], false_disc_std=Float64[], exec_time=Float64[],
               exec_time_std=Float64[], Lambda=Float64[], Mu=Float64[],
               fitted_rank=Float64[], fitted_sparse=Float64[])

successful_entries = 0

for config = 1:NUM_CONFIGS

   file_name = INPUT_PATH * "_" * string(config) * ".json"
   exp_data = Dict()
   open(file_name, "r") do f
      dicttxt = JSON.read(f, String)  # file information to string
      exp_data = JSON.parse(dicttxt)  # parse and transform data
      exp_data = JSON.parse(exp_data)
   end

   num_samples = length(exp_data["execution_time"])
   if num_samples == 0
      continue
   end
   current_row = [exp_data["N"],
                  exp_data["k_rank"],
                  exp_data["k_sparse"],
                  exp_data["signal_to_noise"],
                  Statistics.mean(exp_data["D_error"]),
                  Statistics.std(exp_data["D_error"])/ (num_samples^0.5),
                  Statistics.mean(exp_data["L_error"]),
                  Statistics.std(exp_data["L_error"])/ (num_samples^0.5),
                  Statistics.mean(exp_data["S_error"]),
                  Statistics.std(exp_data["S_error"])/ (num_samples^0.5),
                  Statistics.mean(exp_data["true_discovery"]),
                  Statistics.std(exp_data["true_discovery"])/ (num_samples^0.5),
                  Statistics.mean(exp_data["false_discovery"]),
                  Statistics.std(exp_data["false_discovery"])/ (num_samples^0.5),
                  Statistics.mean(exp_data["execution_time"][2:end]),
                  Statistics.std(exp_data["execution_time"][2:end])/ (num_samples^0.5)]

   if report_lambda_mu

      @assert length(exp_data["Lambda"]) > 0
      @assert length(exp_data["Mu"]) > 0

      append!(current_row, exp_data["Lambda"][1])
      append!(current_row, exp_data["Mu"][1])

   else

      append!(current_row, 0)
      append!(current_row, 0)

   end

   if eval_rank_sparse

      avg_rank = 0
      avg_sparse = 0
      for solution in exp_data["solution"]
         (this_rank, this_sparse) = compute_rank_sparse(solution)
         avg_rank += this_rank
         avg_sparse += this_sparse
      end

      append!(current_row, avg_rank/size(exp_data["solution"])[1])
      append!(current_row, avg_sparse/size(exp_data["solution"])[1])

   else

      append!(current_row, exp_data["k_rank"])
      append!(current_row, exp_data["k_sparse"])

   end

   push!(df, current_row)
   global successful_entries += 1

end

col_names = ["N", "k_rank", "k_sparse", "signal_to_noise"]
attributes = ["_D_error", "_D_error_std", "_L_error", "_L_error_std", "_S_error",
              "_S_error_std", "_true_disc", "_true_disc_std", "_false_disc",
              "_false_disc_std", "_exec_time", "_exec_time_std", "_Lambda",
              "_Mu", "_fitted_rank", "_fitted_sparse"]
for attr in attributes
   push!(col_names, METHOD_NAME * attr)
end
rename!(df, [Symbol(col_names[i]) for i=1:size(col_names)[1]])

if !eval_rank_sparse
   df = select!(df, Not(Symbol(col_names[20])))
   df = select!(df, Not(Symbol(col_names[19])))
end

if !report_lambda_mu
   df = select!(df, Not(Symbol(col_names[18])))
   df = select!(df, Not(Symbol(col_names[17])))
end

CSV.write(OUTPUT_PATH, df)
println("$successful_entries entries have been entered into the dataframe.")
