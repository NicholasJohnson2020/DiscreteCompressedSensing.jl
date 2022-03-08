using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")
using JSON, LinearAlgebra, Statistics, DataFrames, CSV

function processData(input_path, prefix)

   df = DataFrame(damping_factor=Float64[], sample_rate=Float64[],
                  epsilon_multiple=Float64[], L2_error=Float64[],
                  L2_error_std=Float64[], L1_error=Float64[],
                  L1_error_std=Float64[], L0_norm=Float64[],
                  L0_norm_std=Float64[], ssim=Float64[], ssim_std=Float64[],
                  exec_time=Float64[], exec_time_std=Float64[])

   successful_entries = 0

   file_paths = readdir(input_path, join=true)
   for file_name in file_paths

      exp_data = Dict()
      open(file_name, "r") do f
         dicttxt = JSON.read(f, String)
         exp_data = JSON.parse(dicttxt)
         exp_data = JSON.parse(exp_data)
      end

      slice_indexes = exp_data["Slices"]
      L2_error = []
      L1_error = []
      L0_norm = []
      ssim = []
      exec_time = []

      for slice_index in slice_indexes

         append!(L2_error, exp_data[string(slice_index)][prefix * "L2_error"])
         append!(L1_error, exp_data[string(slice_index)][prefix * "L1_error"])
         append!(L0_norm, exp_data[string(slice_index)][prefix * "L0_norm"])
         append!(ssim, exp_data[string(slice_index)][prefix * "ssim"])
         append!(exec_time, exp_data[string(slice_index)][prefix * "execution_time"])

      end

      current_row = [exp_data["Damping"],
                     exp_data["Rate"],
                     exp_data["Epsilon"],
                     Statistics.mean(L2_error),
                     Statistics.std(L2_error),
                     Statistics.mean(L1_error),
                     Statistics.std(L1_error),
                     Statistics.mean(L0_norm),
                     Statistics.std(L0_norm),
                     Statistics.mean(ssim),
                     Statistics.std(ssim),
                     Statistics.mean(exec_time),
                     Statistics.std(exec_time)]
      push!(df, current_row)
      successful_entries += 1

   end

   println("$successful_entries entries have been entered into the dataframe.")
   return df
end;

METHOD_NAME = ARGS[1]
INPUT_PATH = ARGS[2]
numerical_threshold = 1e-4

df1 = processData(INPUT_PATH, "")

if METHOD_NAME in ["BPD_Gurobi_Rounding", "SOC_Relax_Rounding"]
   OUTPUT_PATH = INPUT_PATH * METHOD_NAME * "_rounded_aggrData.csv"
   df2 = processData(INPUT_PATH, "rounded_")
   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_aggrData.csv", df2)
end

CSV.write(INPUT_PATH * METHOD_NAME * "_aggrData.csv", df1)

;
