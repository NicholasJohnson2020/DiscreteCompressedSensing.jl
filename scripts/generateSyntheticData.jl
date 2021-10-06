using Pkg
Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")

using JSON, Random, Distributions

function sample_data(n, p, k, sigma, noise_ratio)

    data_dist = Normal(0, sigma)
    X = rand(data_dist, (n, p))

    beta_temp = rand(data_dist, p)
    indices = randperm(p)[1:k]
    beta = zeros(p)
    for index in indices
        beta[index] = beta_temp[index]
    end

    Y = X * beta
    noise_dist = Normal(0, noise_ratio * sigma)
    noise = rand(noise_dist, n)
    Y = Y + noise

    return X, Y, k, beta

end

file_path = ARGS[1]

NUM_TRIALS_PER_CONFIG = 5
N = collect(50:50:500)
P = collect(10:5:30)
K = collect(2:2:10)
noise_to_signal = collect(0.1:0.05:0.5)

sigma = 5

config_count = 0

data_dict = Dict()
param_dict = Dict()

data_dict["Trials"] = NUM_TRIALS_PER_CONFIG
data_dict["N"] = N
data_dict["P"] = P
data_dict["K"] = K
data_dict["noise_to_signal"] = noise_to_signal
data_dict["sigma"] = sigma

for n in N, p in P, k in K, noise_ratio in noise_to_signal

    global config_count += 1
    param_dict[config_count] = Dict("N"=>n, "P"=>p, "K"=>k,
                                    "noise_ratio"=>noise_ratio)

    current_data_dict = Dict()
    for trial_num = 1:NUM_TRIALS_PER_CONFIG
        X, Y, _, beta = sample_data(n, p, k, sigma, noise_ratio)
        current_data_dict[trial_num] = Dict("X"=>X, "Y"=>Y, "k"=>k,
                                            "beta"=>beta)
    end

    data_dict[string(param_dict[config_count])] = current_data_dict

end

f = open(file_path * "data.json","w")
JSON.print(f, JSON.json(data_dict))
close(f)

f = open(file_path * "params.json","w")
JSON.print(f, JSON.json(param_dict))
close(f)

println("The total number of configs is $config_count.")
