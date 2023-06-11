using JSON, Random, Distributions

function sample_data(m, n, k, sigma, signal_to_noise_ratio)
    """
    This function generates synthetic data according to the data generation
    process defined in Section 6.1.1 of the accompanying paper.

    :param m: The number of observations (Int64).
    :param n: The number of features (Int64).
    :param k: The true sparsity level (Int64).
    :param sigma: The scale of the data (Float64).
    :param signal_to_noise_ratio: The signal to noise ratio (Float64).

    :return: This function returns four values. The first value is the sampled
             m-by-n matrix X, the second is the sampled m vector Y, the third
             is the input specified sparsity level and the fourth is the sampled
             coefficient vector beta.
    """

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

NUM_TRIALS_PER_CONFIG = 100
#N = collect(100:100:800)
N = [50]
#M = collect(100:100:800)
M = [25]
#K = collect(10:5:55)
K = [10]
#ratios = collect(10:2:28)
ratios = [10]
alphas = collect(0.05:0.05:0.9)
#alphas = [0.2]

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

# Main loop to sample data
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

# Store the data to file
f = open(file_path * "data.json","w")
JSON.print(f, JSON.json(data_dict))
close(f)

f = open(file_path * "params.json","w")
JSON.print(f, JSON.json(param_dict))
close(f)

println("The total number of configs is $config_count.")
