using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")
using JSON, LinearAlgebra, Statistics, DataFrames, CSV

NUM_CONFIGS = parse(Int64, ARGS[1])
INPUT_PATH = ARGS[2]

for config = 1:NUM_CONFIGS

   file_name = INPUT_PATH * "_" * string(config) * ".json"
   try
      exp_data = Dict()
      open(file_name, "r") do f
         dicttxt = JSON.read(f, String)  # file information to string
         exp_data = JSON.parse(dicttxt)  # parse and transform data
         exp_data = JSON.parse(exp_data)
      end
   catch
      println("$file_name is not present" )
   end

end
