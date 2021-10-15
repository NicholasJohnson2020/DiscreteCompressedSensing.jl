using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")
using JSON, LinearAlgebra, Statistics, DataFrames, CSV

METHOD_NAME = ARGS[1]
NUM_CONFIGS = parse(Int64, ARGS[2])
INPUT_PATH = ARGS[3]
OUTPUT_PATH = INPUT_PATH * METHOD_NAME * "_aggrData.csv"

df = DataFrame(N=Int64[], p=Int64[], k=Int64[], ratio=Float64[],
               epsilon=Float64[], residual_error=Float64[],
               beta_error=Float64[], fitted_k=Int64[], true_disc=Float64[],
               false_disc=Float64[], exec_time=Float64[])

successful_entries = 0

for config = 1:NUM_CONFIGS

   file_name = INPUT_PATH * "_" * string(config) * ".json"
   exp_data = Dict()
   open(file_name, "r") do f
      dicttxt = JSON.read(f, String)  # file information to string
      exp_data = JSON.parse(dicttxt)  # parse and transform data
      exp_data = JSON.parse(exp_data)
   end

   epsilon_values = exp_dta["epsilon_values"]

   for epsilon in epsilon_values
      current_row = [exp_data["N"],
                     exp_data["P"],
                     exp_data["K"],
                     exp_data["noise_to_signal"],
                     epsilon,
                     Statistics.mean(exp_data[epsilon]["residual_error"]),
                     Statistics.mean(exp_data[epsilon]["beta_error"]),
                     Statistics.mean(exp_data[epsilon]["fitted_k"]),
                     Statistics.mean(exp_data[epsilon]["true_discovery"]),
                     Statistics.mean(exp_data[epsilon]["false_discovery"]),
                     Statistics.mean(exp_data[epsilon]["execution_time"])]
   end

   push!(df, current_row)
   global successful_entries += 1

end

CSV.write(OUTPUT_PATH, df)
println("$successful_entries entries have been entered into the dataframe.")
