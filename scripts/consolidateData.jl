using JSON, LinearAlgebra, Statistics, DataFrames, CSV

function processData(input_path, method_name, prefix="")

   df = DataFrame(N=Int64[], M=Int64[], K=Int64[], signal_to_noise=Float64[],
                  alpha=Float64[], residual_error=Float64[],
                  residual_error_std=Float64[], beta_error=Float64[],
                  beta_error_std=Float64[], fitted_k=Float64[],
                  fitted_k_std=Float64[], true_disc=Float64[],
                  true_disc_std=Float64[], false_disc=Float64[],
                  false_disc_std=Float64[], exec_time=Float64[],
                  exec_time_std=Float64[], lower_bound=Float64[],
                  lower_bound_std=Float64[], root_node_gap=Float64[],
                  root_node_gap_std=Float64[], num_nodes=Float64[],
                  num_nodes_std=Float64[], terminal_nodes=Float64[],
                  terminal_nodes_std=Float64[], epsilon_BnB=Float64[])

   if !(method_name in ["SOC_Relax", "SOC_Relax_Rounded", "SOS"])
      select!(df, Not([:lower_bound, :lower_bound_std]))
   end

   if !(method_name in ["BnB_Primal", "BnB_Dual"])
      select!(df, Not([:root_node_gap, :root_node_gap_std, :num_nodes,
                       :num_nodes_std, :terminal_nodes, :terminal_nodes_std,
                       :epsilon_BnB]))
   end

   successful_entries = 0

   root_path = input_path * method_name * "/"

   file_paths = readdir(root_path, join=true)
   for file_name in file_paths

      exp_data = Dict()
      open(file_name, "r") do f
         dicttxt = JSON.read(f, String)
         exp_data = JSON.parse(dicttxt)
         exp_data = JSON.parse(exp_data)
      end

      num_samples = length(exp_data["execution_time"])
      if num_samples == 0
         continue
      end

      current_row = [exp_data["N"],
                     exp_data["M"],
                     exp_data["K"],
                     exp_data["signal_to_noise"],
                     exp_data["alpha"],
                     Statistics.mean(exp_data[prefix * "residual_error"]),
                     Statistics.std(exp_data[prefix * "residual_error"]) / (num_samples^0.5),
                     Statistics.mean(exp_data[prefix * "beta_error"]),
                     Statistics.std(exp_data[prefix * "beta_error"]) / (num_samples^0.5),
                     Statistics.mean(exp_data[prefix * "fitted_k"]),
                     Statistics.std(exp_data[prefix * "fitted_k"]) / (num_samples^0.5),
                     Statistics.mean(exp_data[prefix * "true_discovery"]),
                     Statistics.std(exp_data[prefix * "true_discovery"]) / (num_samples^0.5),
                     Statistics.mean(exp_data[prefix * "false_discovery"]),
                     Statistics.std(exp_data[prefix * "false_discovery"]) / (num_samples^0.5),
                     Statistics.mean(exp_data[prefix * "execution_time"][2:end]),
                     Statistics.std(exp_data[prefix * "execution_time"][2:end]) / (num_samples^0.5)]

      if method_name in ["SOC_Relax", "SOC_Relax_Rounded", "SOS"]
         append!(current_row, Statistics.mean(exp_data["lower_bound"]))
         append!(current_row, Statistics.std(exp_data["lower_bound"]) / (num_samples^0.5))
      end

      if method_name in ["BnB_Primal", "BnB_Dual"]
         append!(current_row, Statistics.mean(exp_data["root_node_gap"]))
         append!(current_row, Statistics.std(exp_data["root_node_gap"]) / (num_samples^0.5))
         append!(current_row, Statistics.mean(exp_data["num_nodes"]))
         append!(current_row, Statistics.std(exp_data["num_nodes"]) / (num_samples^0.5))
         append!(current_row, Statistics.mean(exp_data["terminal_nodes"]))
         append!(current_row, Statistics.std(exp_data["terminal_nodes"]) / (num_samples^0.5))
         append!(current_row, exp_data["epsilon_BnB"][1])
      end

      push!(df, current_row)
      successful_entries += 1

   end

   println("$successful_entries entries have been entered into the dataframe.")
   return df
end;


METHOD_NAME = ARGS[1]
INPUT_PATH = ARGS[2]

OUTPUT_ROOT = INPUT_PATH * METHOD_NAME * "/" * METHOD_NAME

df1 = processData(INPUT_PATH, METHOD_NAME, "")

if METHOD_NAME in ["BPD_Rounded", "IRWL1_Rounded", "SOC_Relax_Rounded"]
   df2 = processData(INPUT_PATH, METHOD_NAME, "rounded_")
   CSV.write(OUTPUT_ROOT * "_rounded_aggrData.csv", df2)
end

CSV.write(OUTPUT_ROOT * "_aggrData.csv", df1)
