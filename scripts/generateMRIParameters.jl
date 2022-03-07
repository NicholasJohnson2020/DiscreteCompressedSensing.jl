using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")

using JSON

file_path = ARGS[1]

#DAMPING_FACTOR = collect(2:2:10)
DAMPING_FACTOR = [4]
SAMPLE_RATE = collect(0.2:0.1:0.9)
EPSILON_MULTIPLE = collect(1.5:0.5:5)

config_count = 0

param_dict = Dict()

for factor in DAMPING_FACTOR, rate in SAMPLE_RATE, epsilon in EPSILON_MULTIPLE

    global config_count += 1
    param_dict[config_count] = Dict("DAMPING_FACTOR"=>factor,
                                    "SAMPLE_RATE"=>rate,
                                    "EPSILON_MULTIPLE"=>epsilon)

end

f = open(file_path * "params.json","w")
JSON.print(f, JSON.json(param_dict))
close(f)

println("The total number of configs is $config_count.")
