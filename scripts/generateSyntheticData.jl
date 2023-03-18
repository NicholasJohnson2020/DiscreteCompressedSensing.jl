using JSON, Random, Distributions

function sample_data(m, n, k, sigma, signal_to_noise_ratio)

    data_dist = Normal(0, signal_to_noise_ratio * sigma / n^0.5)
    X = rand(data_dist, (m, n))

    beta_temp = rand(data_dist, n)
    indices = randperm(n)[1:k]
    beta = zeros(n)
    for index in indices
        beta[index] = beta_temp[index]
    end

    Y = X * beta
    noise_dist = Normal(0, sigma)
    noise = rand(noise_dist, m)
    Y = Y + noise

    return X, Y, k, beta

end

file_path = ARGS[1]

NUM_TRIALS_PER_CONFIG = 50
N = collect(100:100:1000)
#N = [200]
#M = collect(100:100:1000)
M = [100]
#K = collect(5:5:50)
K = [10]
#ratios = collect(2:2:30)
ratios = [10]
# alphas = collect(0.05:0.05:0.95)
alphas = [0.5]

sigma = 1

config_count = 0

data_dict = Dict()
param_dict = Dict()

data_dict["Trials"] = NUM_TRIALS_PER_CONFIG
data_dict["N"] = N
data_dict["P"] = M
data_dict["K"] = K
data_dict["signal_to_noise"] = ratios
data_dict["alpha"] = alphas
data_dict["sigma"] = sigma

for n in N, m in M, k in K, signal_to_noise in ratios, alpha in alphas

    global config_count += 1
    param_dict[config_count] = Dict("N"=>n, "M"=>m, "K"=>k,
                                    "signal_ratio"=>signal_to_noise,
                                    "alpha"=>alpha)

    current_data_dict = Dict()
    for trial_num = 1:NUM_TRIALS_PER_CONFIG
        X, Y, _, beta = sample_data(m, n, k, sigma, signal_to_noise)
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
