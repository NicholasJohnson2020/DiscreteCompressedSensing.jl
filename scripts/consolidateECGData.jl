using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")
using JSON, LinearAlgebra, Statistics, DataFrames, CSV

function processData(input_path, prefix)

   df = DataFrame(epsilon_multiple=Float64[], M=Int64[], CR=Float64[],
                  L2_error=Float64[], L2_error_std=Float64[],
                  L1_error=Float64[], L1_error_std=Float64[],
                  L0_norm=Float64[], L0_norm_std=Float64[],
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

      patient_indexes = exp_data["Indices"]
      L2_error = []
      L1_error = []
      L0_norm = []
      exec_time = []

      for patientID in patient_indexes

         append!(L2_error, exp_data[string(patientID)][prefix * "L2_error"])
         append!(L1_error, exp_data[string(patientID)][prefix * "L1_error"])
         append!(L0_norm, exp_data[string(patientID)][prefix * "L0_norm"])
         append!(exec_time, exp_data[string(patientID)][prefix * "execution_time"])

      end

      current_row = [exp_data["EPSILON"],
                     exp_data["M"],
                     exp_data["CR"],
                     Statistics.mean(L2_error),
                     Statistics.std(L2_error),
                     Statistics.mean(L1_error),
                     Statistics.std(L1_error),
                     Statistics.mean(L0_norm),
                     Statistics.std(L0_norm),
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

if METHOD_NAME == "BPD_Gurobi_Rounding"
   OUTPUT_PATH = INPUT_PATH * METHOD_NAME * "_rounded_aggrData.csv"
   df2 = processData(INPUT_PATH, "rounded_")
   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_aggrData.csv", df2)
end

if METHOD_NAME == "SOC_Relax_Rounding"
   OUTPUT_PATH = INPUT_PATH * METHOD_NAME * "_rounded_x_aggrData.csv"
   df2 = processData(INPUT_PATH, "rounded_x_")
   OUTPUT_PATH = INPUT_PATH * METHOD_NAME * "_rounded_z_aggrData.csv"
   df3 = processData(INPUT_PATH, "rounded_z_")
   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_x_aggrData.csv", df2)
   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_z_aggrData.csv", df3)
end

CSV.write(INPUT_PATH * METHOD_NAME * "_aggrData.csv", df1)

;