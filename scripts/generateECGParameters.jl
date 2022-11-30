using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")

using JSON

file_path = ARGS[1]

#NUM_CR = collect(1:2:19)
#EPSILON_MULTIPLE = collect(0.05:0.05:0.5)
#M_vals = collect(10:5:45)
M_vals = [50]
#EPSILON_MULTIPLE = collect(1:0.05:1.5)
EPSILON_MULTIPLE = [1.05]
N = 1024
#GAMMA_MULT = [0.01, 0.1, 1, 10]
GAMMA_MULT = collect(LinRange(0.01, 8, 160))
GAMMA_FLAG = ["SQUARE_ROOT", "LINEAR", "SQUARE"]

config_count = 0

param_dict = Dict()

#for num in NUM_CR, eps in EPSILON_MULTIPLE
for gamma_flag in GAMMA_FLAG
    for gamma_mult in GAMMA_MULT
        for M in M_vals, eps in EPSILON_MULTIPLE

            global config_count += 1
            #M = Int64(floor((1-num*5*0.01)*N))
            CR = 1 - M/N
            param_dict[config_count] = Dict("M"=>M,
                                            "CR"=>CR,
                                            "EPSILON"=>eps,
                                            "GAMMA_MULT"=>gamma_mult,
                                            "GAMMA_FLAG"=>gamma_flag)

        end
    end
end

f = open(file_path * "params.json","w")
JSON.print(f, JSON.json(param_dict))
close(f)

println("The total number of configs is $config_count.")
