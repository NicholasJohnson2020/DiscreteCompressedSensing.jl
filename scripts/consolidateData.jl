using JSON, LinearAlgebra, Statistics, DataFrames, CSV

function computeStats(beta_true, beta_fitted, numerical_threshold=1e-4)
   """
   This function computes performance metrics for the vector beta_fitted
   relative to the vector beta_true when viewed as a classification problem
   to select nonzero entries.

   :param beta_true: The ground truth.
   :param beta_fitted: An estimate of beta_true.
   :param numerical_threshold: The threshold above which an entry in beta_fitted
                               is considered to be nonzero.

   :return: This function returns three values. The first is the true positive
            rate of beta_fitted, the second is the true negative rate of
            beta_fitted, the third is the accuracy of beta_fitted.
   """
   pos_fitted_indices = abs.(beta_fitted) .> numerical_threshold
   neg_fitted_indices = .~pos_fitted_indices
   pos_true_indices = abs.(beta_true) .> numerical_threshold
   neg_true_indices = .~pos_true_indices

   TP = sum(abs.(beta_fitted[pos_true_indices]) .> numerical_threshold)
   P = sum(pos_fitted_indices)
   TPR = TP / P

   TN = sum(abs.(beta_fitted[neg_true_indices]) .< numerical_threshold)
   N = sum(neg_fitted_indices)
   TNR = TN / N

   ACC = (TP + TN) / (P + N)

   return TPR, TNR, ACC

end;

function processData(input_path, method_name, prefix="")
   """
   This function loads raw experiment output data and processes it into a
   dataframe.
   """
   df = DataFrame(N=Int64[], M=Int64[], K=Int64[], signal_to_noise=Float64[],
                  alpha=Float64[], residual_error=Float64[],
                  residual_error_std=Float64[], beta_error=Float64[],
                  beta_error_std=Float64[], fitted_k=Float64[],
                  fitted_k_std=Float64[], true_disc=Float64[],
                  true_disc_std=Float64[], false_disc=Float64[],
                  false_disc_std=Float64[], exec_time=Float64[],
                  exec_time_std=Float64[], TPR=Float64[], TPR_std=Float64[],
                  TNR=Float64[], TNR_std=Float64[], ACC=Float64[],
                  ACC_std=Float64[], lower_bound=Float64[],
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
   # Iterate over all files in the input directory
   for file_name in file_paths

      exp_data = Dict()
      open(file_name, "r") do f
         dicttxt = JSON.read(f, String)
         exp_data = JSON.parse(dicttxt)
         exp_data = JSON.parse(exp_data)
      end

      task_ID = file_name[end-6:end-5]
      if task_ID[1] == '_'
          task_ID = task_ID[2:end]
      end;

      key = Dict{String, Real}(param_dict[task_ID])
      experiment_data = synthetic_data[string(key)]

      num_samples = length(exp_data["execution_time"])
      if num_samples == 0
         continue
      end

      # Extract and store the relevant data
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

      TPR_vec = zeros(num_samples)
      TNR_vec = zeros(num_samples)
      ACC_vec = zeros(num_samples)
      for trial_num=1:num_samples
         beta_fitted = exp_data[prefix * "solution"][trial_num]
         beta_true = experiment_data[string(trial_num)]["beta"]
         TPR, TNR, ACC = computeStats(beta_true, beta_fitted)
         TPR_vec[trial_num] = TPR
         TNR_vec[trial_num] = TNR
         ACC_vec[trial_num] = ACC
      end

      temp_row = [Statistics.mean(TPR_vec),
                  Statistics.std(TPR_vec) / (num_samples^0.5),
                  Statistics.mean(TNR_vec),
                  Statistics.std(TNR_vec) / (num_samples^0.5),
                  Statistics.mean(ACC_vec),
                  Statistics.std(ACC_vec) / (num_samples^0.5)]

      current_row = vcat(current_row, temp_row)

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

# Load the synthetic data
synthetic_data = Dict()
open(INPUT_PATH * "SynExp_data.json", "r") do f
    global synthetic_data
    dicttxt = JSON.read(f, String)  # file information to string
    synthetic_data = JSON.parse(dicttxt)  # parse and transform data
    synthetic_data = JSON.parse(synthetic_data)
end

# Load the experiment parameters
param_dict = Dict()
open(INPUT_PATH * "SynExp_params.json", "r") do f
    global param_dict
    dicttxt = JSON.read(f, String)  # file information to string
    param_dict = JSON.parse(dicttxt)  # parse and transform data
    param_dict = JSON.parse(param_dict)
end

# Process and save the data
df1 = processData(INPUT_PATH, METHOD_NAME, "")

if METHOD_NAME in ["BPD_Rounded", "IRWL1_Rounded", "SOC_Relax_Rounded"]
   df2 = processData(INPUT_PATH, METHOD_NAME, "rounded_")
   CSV.write(OUTPUT_ROOT * "_rounded_aggrData.csv", df2)
end

CSV.write(OUTPUT_ROOT * "_aggrData.csv", df1)
