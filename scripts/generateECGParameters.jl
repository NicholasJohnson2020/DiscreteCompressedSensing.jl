using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")

using JSON

file_path = ARGS[1]

#NUM_CR = collect(1:2:19)
#EPSILON_MULTIPLE = collect(0.05:0.05:0.5)
M_vals = collect(5:5:50)
EPSILON_MULTIPLE = collect(1:0.05:1.5)
N = 1024

config_count = 0

param_dict = Dict()

#for num in NUM_CR, eps in EPSILON_MULTIPLE
for M in M_vals, eps in EPSILON_MULTIPLE

    global config_count += 1
    #M = Int64(floor((1-num*5*0.01)*N))
    CR = 1 - M/N
    param_dict[config_count] = Dict("M"=>M,
                                    "CR"=>CR,
                                    "EPSILON"=>eps)

end

f = open(file_path * "params.json","w")
JSON.print(f, JSON.json(param_dict))
close(f)

println("The total number of configs is $config_count.")
