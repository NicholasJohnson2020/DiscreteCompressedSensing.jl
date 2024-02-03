using JSON, LinearAlgebra, Statistics, DataFrames, CSV

numerical_threshold = 1e-4

function processData(input_path, prefix; BnB=false)
   """
   This function loads raw experiment output data and processes it into a
   dataframe.
   """
   df = DataFrame(index=Any[], epsilon_multiple=Any[],
                  gamma_mult=Any[], gamma_flag=Any[], L2_error=Any[],
                  L2_error_rel=Any[], L0_norm=Any[], TPR=Any[],
                  TNR=Any[], acc=Any[], precision=Any[], exec_time=Any[])

   successful_entries = 0

   file_paths = readdir(input_path, join=true)
   num_nodes = []

   # Iterate over all files in the input directory
   for file_name in file_paths

      exp_data = Dict()
      open(file_name, "r") do f
         dicttxt = JSON.read(f, String)
         exp_data = JSON.parse(dicttxt)
         exp_data = JSON.parse(exp_data)
      end

      if sum(exp_data["b_full"]) == 0.0
         continue
      end

      true_ind = findall(==(1), exp_data["b_full"])
      selected_ind = findall(==(1), abs.(exp_data[prefix * "solution"]) .> numerical_threshold)
      precision = size(findall(in(true_ind), selected_ind))[1] / exp_data[prefix * "L0_norm"]

      # Extract and store the relevant data
      current_row = [exp_data["TASK_ID"],
                     exp_data["EPSILON"],
                     exp_data["GAMMA_MULT"],
                     exp_data["GAMMA_FLAG"],
                     exp_data[prefix * "L2_error"],
                     exp_data[prefix * "L2_error_rel"],
                     exp_data[prefix * "L0_norm"],
                     exp_data[prefix * "TPR"],
                     exp_data[prefix * "TNR"],
                     exp_data[prefix * "accuracy"],
                     precision
                     exp_data[prefix * "execution_time"]]

      if BnB
         append!(num_nodes, exp_data["num_nodes"])
      end
      push!(df, current_row)
      successful_entries += 1

   end

   if BnB
      df[!, "num_nodes"] = num_nodes
   end

   println("$successful_entries entries have been entered into the dataframe.")
   return df
end;

METHOD_NAME = ARGS[1]
INPUT_PATH = ARGS[2] * METHOD_NAME * "/"

# Process and save the data
if METHOD_NAME in ["BnB_Primal", "BnB_Primal_2"]
   df1 = processData(INPUT_PATH, "", BnB = true)
else
   df1 = processData(INPUT_PATH, "")
end

if METHOD_NAME in ["BPD_Rounded", "IRWL1_Rounded"]
   df2 = processData(INPUT_PATH, "rounded_")
   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_aggrData.csv", df2)
end

#if METHOD_NAME == "SOC_Relax_Rounded"
#   df2 = processData(INPUT_PATH, "rounded_x_")
#   df3 = processData(INPUT_PATH, "rounded_z_")
#   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_x_aggrData.csv", df2)
#   CSV.write(INPUT_PATH * METHOD_NAME * "_rounded_z_aggrData.csv", df3)
#end

CSV.write(INPUT_PATH * METHOD_NAME * "_aggrData.csv", df1)
;
