using JSON

file_path = ARGS[1]

#GAMMA_MULT = [0.01, 0.1, 1, 10]
GAMMA_MULT = collect(LinRange(0.01, 8, 120))
GAMMA_FLAG = ["SQUARE_ROOT", "LINEAR", "SQUARE"]

config_count = 0

param_dict = Dict()

# Main loop to generate parameters
for gamma_flag in GAMMA_FLAG
    for gamma_mult in GAMMA_MULT

        global config_count += 1
        param_dict[config_count] = Dict("GAMMA_MULT"=>gamma_mult,
                                        "GAMMA_FLAG"=>gamma_flag)

    end
end

# Save parameters to file
f = open(file_path * "params.json","w")
JSON.print(f, JSON.json(param_dict))
close(f)

println("The total number of configs is $config_count.")
