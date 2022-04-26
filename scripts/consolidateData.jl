using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")
using JSON, LinearAlgebra, Statistics, DataFrames, CSV

function processData(input_path, prefix)

   df = DataFrame(N=Int64[], p=Int64[], k=Int64[], ratio=Float64[],
                  epsilon=Float64[], residual_error=Float64[],
                  beta_error=Float64[], fitted_k=Float64[], true_disc=Float64[],
                  false_disc=Float64[], exec_time=Float64[])

   successful_entries = 0

   file_paths = readdir(input_path, join=true)
   for file_name in file_paths

      exp_data = Dict()
      open(file_name, "r") do f
         dicttxt = JSON.read(f, String)
         exp_data = JSON.parse(dicttxt)
         exp_data = JSON.parse(exp_data)
      end

      epsilon_values = exp_data["epsilon_values"]

      for epsilon in epsilon_values
         fitted_k_vals = []
         for sol in exp_data[string(epsilon)]["solution"]
            fitted_k = sum(abs.(sol) .> numerical_threshold)
            append!(fitted_k_vals, fitted_k)
         end
         current_row = [exp_data["N"],
                        exp_data["P"],
                        exp_data["K"],
                        exp_data["noise_to_signal"],
                        epsilon,
                        Statistics.mean(exp_data[string(epsilon)]["residual_error"]),
                        Statistics.mean(exp_data[string(epsilon)]["beta_error"]),
                        #Statistics.mean(exp_data[string(epsilon)]["fitted_k"]),
                        Statistics.mean(fitted_k_vals),
                        Statistics.mean(exp_data[string(epsilon)]["true_discovery"]),
                        Statistics.mean(exp_data[string(epsilon)]["false_discovery"]),
                        Statistics.mean(exp_data[string(epsilon)]["execution_time"])]
         push!(df, current_row)
         successful_entries += 1

      end

   end

   println("$successful_entries entries have been entered into the dataframe.")
   return df
end;


METHOD_NAME = ARGS[1]
INPUT_PATH = ARGS[2]

numerical_threshold = 1e-4

df1 = processData(INPUT_PATH, "")

if METHOD_NAME == "BPD_Gurobi_Rounding"
   df2 = processData(INPUT_PATH, "rounded_")
   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_aggrData.csv", df2)
end

if METHOD_NAME == "SOC_Relax_Rounding"
   df2 = processData(INPUT_PATH, "rounded_x_")
   df3 = processData(INPUT_PATH, "rounded_z_")
   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_x_aggrData.csv", df2)
   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_z_aggrData.csv", df3)
end

CSV.write(INPUT_PATH * METHOD_NAME * "_aggrData.csv", df1)
