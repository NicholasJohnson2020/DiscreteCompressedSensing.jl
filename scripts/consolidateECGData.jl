using JSON, LinearAlgebra, Statistics, DataFrames, CSV

function processData(input_path, prefix; BnB=false)

   df = DataFrame(epsilon_multiple=Float64[], M=Int64[], CR=Float64[],
                  gamma_mult=Float64[], gamma_flag=String[],
                  L2_error=Float64[], L2_error_std=Float64[],
                  L1_error=Float64[], L1_error_std=Float64[],
                  L0_norm=Float64[], L0_norm_std=Float64[],
                  exec_time=Float64[], exec_time_std=Float64[])

   successful_entries = 0

   file_paths = readdir(input_path, join=true)
   num_nodes_mean = []
   num_nodes_std = []

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

      num_nodes = []

      for patientID in patient_indexes

         append!(L2_error, exp_data[string(patientID)][prefix * "L2_error"])
         append!(L1_error, exp_data[string(patientID)][prefix * "L1_error"])
         append!(L0_norm, exp_data[string(patientID)][prefix * "L0_norm"])
         append!(exec_time, exp_data[string(patientID)][prefix * "execution_time"])

         if BnB
            append!(num_nodes, exp_data[string(patientID)]["num_nodes"])
         end

      end

      current_row = [exp_data["EPSILON"],
                     exp_data["M"],
                     exp_data["CR"],
                     exp_data["GAMMA_MULT"],
                     exp_data["GAMMA_FLAG"],
                     Statistics.mean(L2_error),
                     Statistics.std(L2_error),
                     Statistics.mean(L1_error),
                     Statistics.std(L1_error),
                     Statistics.mean(L0_norm),
                     Statistics.std(L0_norm),
                     Statistics.mean(exec_time),
                     Statistics.std(exec_time)]

      if BnB
         append!(num_nodes_mean, Statistics.mean(num_nodes))
         append!(num_nodes_std, Statistics.std(num_nodes))
      end
      push!(df, current_row)
      successful_entries += 1

   end

   if BnB
      df[!, "num_nodes"] = num_nodes_mean
      df[!, "num_nodes_error"] = num_nodes_std
   end

   println("$successful_entries entries have been entered into the dataframe.")
   return df
end;

METHOD_NAME = ARGS[1]
INPUT_PATH = ARGS[2] * METHOD_NAME * "/"
numerical_threshold = 1e-4

if METHOD_NAME in ["BnB_Primal"]
   df1 = processData(INPUT_PATH, "", BnB = true)
else
   df1 = processData(INPUT_PATH, "")
end

if METHOD_NAME in ["BPD_Gurobi_Rounded", "IRWL1_Rounded"]
   OUTPUT_PATH = INPUT_PATH * METHOD_NAME * "_rounded_aggrData.csv"
   df2 = processData(INPUT_PATH, "rounded_")
   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_aggrData.csv", df2)
end

if METHOD_NAME == "SOC_Relax_Rounded"
   OUTPUT_PATH = INPUT_PATH * METHOD_NAME * "_rounded_x_aggrData.csv"
   df2 = processData(INPUT_PATH, "rounded_x_")
   OUTPUT_PATH = INPUT_PATH * METHOD_NAME * "_rounded_z_aggrData.csv"
   df3 = processData(INPUT_PATH, "rounded_z_")
   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_x_aggrData.csv", df2)
   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_z_aggrData.csv", df3)
end

CSV.write(INPUT_PATH * METHOD_NAME * "_aggrData.csv", df1)
;
