include("helperLibrary.jl")
include("subProblems.jl")

struct node
    zero_indices::Array{Any, 1}
    one_indices::Array{Any, 1}
    upper_bound::Float64
    lower_bound::Float64
    relaxed_binary_vector::Array{Float64, 1}
end

function isTerminal(one_indices, zero_indices, x_dim)
    """
    This function evaluates whether the current node in the Branch and Bound
    tree is a terminal node.

    :param zero_indices: List of 2-tuples of Int64 where each entry consists of
                         an index (i, j) such that Y_ij is constrained to take
                         value 0.
    :param one_indices: List of 2-tuples of Int64 where each entry consists of
                        an index (i, j) such that S_ij is constrained to take
                        value 1 where S is the binary matrix denoting the
                        sparsity pattern of Y.
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param matrix_dim: The row and column dimension of the data (Int64).

    :return: True if this is a terminal node, false otherwise (Bool).
    """

    num_fixed_ones = size(one_indices)[1]
    num_fixed_zeros = size(zero_indices)[1]

    if num_fixed_ones + num_fixed_zeros == x_dim
        return true
    end

    return false

end

function solveSubproblem(A, b, epsilon, gamma; round_solution=false,
    zero_indices=[], one_indices=[], subproblem_type="primal")

    output = nothing
    if subproblem_type == "primal"
        output = solveSubproblemPrimal(A, b, epsilon, gamma,
                                       round_solution=round_solution,
                                       zero_indices=zero_indices,
                                       one_indices=one_indices)
    elseif subproblem_type == "dual"
        opt_val, opt_nu = solveSubproblemDual(A, b, epsilon, gamma,
                                              zero_indices=zero_indices,
                                              one_indices=one_indices)
        z_relax = float.(A'*opt_nu .> 0) - A'*opt_nu * sqrt(gamma) / 4
        for index in zero_indices
            z_relax[index] = 0
        end
        for index in one_indices
            z_relax[index] = 1
        end
        if round_solution
            rounded_x, num_support = roundSolution(z_relax, A, b, epsilon)
            output = (rounded_x, num_support, nothing, z_relax, opt_val)
        else
            z_relax = float.(A'*opt_nu .> 0) - A'*opt_nu * sqrt(gamma) / 4
            output = (nothing, z_relax, opt_val)
        end
    end

    return output
end;

function CS_BnB(A, b, epsilon, gamma; termination_threshold=0.1,
    round_at_nodes=false, output_to_file=false, output_file_name="temp.txt",
    subproblem_type="primal", subproblem_warmstart=false, BPD_backbone=false,
    use_default_gamma=false, cutoff_time=5)
    """
    This function computes a certifiably near-optimal solution to the problem
        min ||U - X - Y||_F^2 + lambda * ||X||_F^2 + mu * ||Y||_F^2
        subject to rank(X) <= k_rank, ||Y||_0 <= k_sparse

    by executing a custom branch and bound algorithm.

    :param U: An arbitrary n-by-n matrix.
    :param mu: The regularization parameter for the sparse matrix penalty
                   (Float64).
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).
    :param epsilon: The Branch and Bound termination criterion. This function
                    will terminate with an epsilon optimal solution to the
                    optimization problem (Float64).
    :param iter_min_improv: The minimal fractional decrease in the objective
                            value required for the alternating minimization
                            procedure to continue iterating.
    :param symmetric_tree: If true, Branch and Bound will only explore symmetric
                           sparsity patterns (Bool).
    :param output_to_file: If true, progress output will be printed to file. If
                           false, progress output will be printed to the console
                           (Bool).
    :param output_file_name: Name of the file progress output will be printed to
                             when output_to_file=true (String).

    :return: This function returns eight values:
             1) A tuple of two n-by-n arrays that correspond to the globally
                optimal solution to the optimization problem. The first element
                in the tuple is the optimal low rank matrix and the second
                element in the tuple is the optimal sparse matrix.
             2) The final global upper bound (Float64).
             3) The final global lower bound (Float64).
             4) The number of nodes explored during the optimization process
                (Int64).
             5) A list of values of type Float64 representing the evolution of
                the upper bound during the optimization procedure.
             6) A list of values of type Float64 representing the evolution of
                the lower bound during the optimization procedure.
             7) The total time elapsed during the optimiation process
                (milliseconds).
             8) The number of terminal nodes explored during the optimization
                process (Int64).

                best feasible solution found
             by this method (the first element in the tuple is the matrix X and
             the second element is the matrix Y). The second value is the
             objective value (Float64) achieved by the returned feasible
             solution.
    """

    @assert subproblem_type in ["primal", "dual"]
    @assert round_at_nodes in [false, true]
    @assert subproblem_warmstart in [false, true]

    start_time = now()

    A = convert(Array{Float64,2}, A)
    b = convert(Array{Float64,1}, b)

    if norm(A * (A \ b) - b)^2 > epsilon
        println("Problem is infeasible.")
        return false
    end

    # Solve the root node
    (m, n) = size(A)
    original_n = n
    backbone = []

    init_zero_indices = []
    init_one_indices = []

    output = basisPursuitDenoising(A, b, epsilon, round_solution=true)

    num_support = output[1]
    rounded_x = output[2]
    opt_x = output[4]

    if BPD_backbone
        for index=1:size(opt_x)[1]
            if abs(opt_x[index]) > 1e-6
                append!(backbone, index)
            end
        end
        n = size(backbone)[1]
        reduced_A = zeros(m, n)
        for i=1:n
            reduced_A[:, i] = A[:, backbone[i]]
        end
        A = reduced_A

        rounded_x = zeros(n)
        opt_x = zeros(n)
        for i=1:n
            rounded_x[i] = output[2][backbone[i]]
            opt_x[i] = output[4][backbone[i]]
        end
    end

    if use_default_gamma
        gamma = sqrt(size(A)[2])
    end

    all_indices = collect(1:n)

    upper_bound_sol = rounded_x
    upper_bound_obj = num_support + norm(rounded_x)^2 / gamma

    lower_bound_sol_x = opt_x
    lower_bound_sol_z = abs.(opt_x) ./ sqrt(gamma)
    lower_bound_obj = sum(abs.(opt_x)) * 2 / sqrt(gamma)

    root_optimality_gap = (upper_bound_obj - lower_bound_obj) / upper_bound_obj
    if output_to_file
        open(output_file_name, "a") do io
            write(io, "Root Node has been solved.\n")
            write(io, "Root Node upper bound is: $upper_bound_obj\n")
            write(io, "Root Node lower bound is: $lower_bound_obj\n")
            write(io, "Root Node optimality gap is: $root_optimality_gap\n\n")
        end
    else
        println("Root Node has been solved.")
        println("Root Node upper bound is: $upper_bound_obj")
        println("Root Node lower bound is: $lower_bound_obj")
        println("Root Node optimality gap is: $root_optimality_gap")
        println()
    end

    # Initialize the node list and global bounds
    master_node_list = []
    root_node = node(init_zero_indices,
                     init_one_indices,
                     upper_bound_obj,
                     lower_bound_obj,
                     lower_bound_sol_z)
    push!(master_node_list, root_node)
    global_upper_bound = upper_bound_obj
    global_lower_bound = lower_bound_obj
    global_x_sol = upper_bound_sol

    num_explored_nodes = 1
    upper_bound_hist = [global_upper_bound]
    lower_bound_hist = [global_lower_bound]

    terminal_nodes = 0
    infeasible_sets = []
    # Main branch and bound loop
    while (global_upper_bound - global_lower_bound) /
                                    global_upper_bound > termination_threshold

        """
        # Select the most recently added node (DFS)
        current_node = pop!(master_node_list)
        """
        current_node_index = findall(this_node->
            this_node.lower_bound == global_lower_bound, master_node_list)[1]
        current_node = master_node_list[current_node_index]
        deleteat!(master_node_list, current_node_index)

        # If the node is a terminal node, place it back on the master list
        if isTerminal(current_node.one_indices, current_node.zero_indices, n)
            prepend!(master_node_list, current_node)
            continue
        end
        # Select entry to branch on using most fractional rule
        # Will later change this to utilize strong branching
        test_vector = abs.(current_node.relaxed_binary_vector .- 0.5)
        test_vector[current_node.one_indices] .= 10
        test_vector[current_node.zero_indices] .= 10
        index = argmin(test_vector)

        # Construct the two child sets of indices from this parent node
        new_index_zero = (append!(copy(current_node.zero_indices), index),
                          copy(current_node.one_indices))
        new_index_one = (copy(current_node.zero_indices),
                         append!(copy(current_node.one_indices), index))
        new_index_list = [new_index_zero, new_index_one]


        for (zero_indices, one_indices) in new_index_list

            this_set = Set(zero_indices)
            for that_set in infeasible_sets
                if issubset(that_set, this_set)
                    continue
                end
            end
            pos_ind = all_indices[(!in).(all_indices, Ref(zero_indices))]
            if norm(A[:, pos_ind] * (A[:, pos_ind] \ b) - b) ^2 > epsilon
                append!(infeasible_sets, this_set)
                continue
            end
            upper_bound_obj = global_upper_bound
            output = solveSubproblem(A, b, epsilon, gamma,
                                     round_solution=round_at_nodes,
                                     zero_indices=zero_indices,
                                     one_indices=one_indices,
                                     subproblem_type=subproblem_type)
            if round_at_nodes
                (rounded_x, upper_bound_obj, x_relax, z_relax, lower_bound_obj) = output
            else
                (x_relax, z_relax, lower_bound_obj) = output
            end

            if isTerminal(one_indices, zero_indices, n) || round_at_nodes
                if isTerminal(one_indices, zero_indices, n)
                    upper_bound_obj = lower_bound_obj
                    if x_relax == nothing
                        output = solveSubproblem(A, b, epsilon, gamma,
                                                 round_solution=false,
                                                 zero_indices=zero_indices,
                                                 one_indices=one_indices,
                                                 subproblem_type="primal")
                        upper_bound_sol = output[1]
                    else
                        upper_bound_sol = x_relax
                    end
                else
                    upper_bound_sol = rounded_x
                end

                if upper_bound_obj < global_upper_bound
                    global_upper_bound = upper_bound_obj
                    global_x_sol = upper_bound_sol

                    current_gap = global_upper_bound - global_lower_bound
                    current_gap = current_gap / global_upper_bound

                    println("The new upper bound is: $global_upper_bound")
                    println("The current lower bound is: $global_lower_bound")
                    println("The current optimality gap is: $current_gap")
                    println()

                    if output_to_file
                        open(output_file_name, "a") do io
                            write(io, "The new upper bound is: $global_upper_bound\n")
                            write(io, "The current lower bound is: $global_lower_bound\n")
                            write(io, "The current optimality gap is: $current_gap\n\n")
                        end
                    else
                        println("The new upper bound is: $global_upper_bound")
                        println("The current lower bound is: $global_lower_bound")
                        println("The current optimality gap is: $current_gap")
                        println()
                    end

                    new_master_node_list = []
                    for this_node in master_node_list
                        if this_node.lower_bound < global_upper_bound
                            push!(new_master_node_list, this_node)
                        end
                    end

                    pre_size = size(master_node_list)[1] + 1
                    post_size = size(new_master_node_list)[1] + 1

                    if output_to_file
                        open(output_file_name, "a") do io
                            write(io, "Tree size before pruning: $pre_size\n")
                            write(io, "Tree size after pruning: $post_size\n\n")
                        end
                    else
                        println("Tree size before pruning: $pre_size")
                        println("Tree size after pruning: $post_size")
                        println()
                    end

                    master_node_list = new_master_node_list
                end
            end

            if lower_bound_obj < global_upper_bound
                new_node = node(zero_indices,
                                one_indices,
                                upper_bound_obj,
                                lower_bound_obj,
                                z_relax)
                push!(master_node_list, new_node)
            end

            num_explored_nodes = num_explored_nodes + 1
            append!(upper_bound_hist, global_upper_bound)
            append!(lower_bound_hist, global_lower_bound)

            if num_explored_nodes % 50 == 0
                current_time = now() - start_time

                if output_to_file
                    open(output_file_name, "a") do io
                        write(io, "$num_explored_nodes nodes have been explored.\n")
                        write(io, "Current elapsed time: $current_time\n")
                        write(io, "The current lower bound is: $global_lower_bound\n")
                    end
                else
                    println("$num_explored_nodes nodes have been explored.")
                    println("Current elapsed time: $current_time")
                    println("The current lower bound is: $global_lower_bound")
                end
            end

        end

        if size(master_node_list)[1] == 0
            if output_to_file
                open(output_file_name, "a") do io
                    write(io, "All nodes have been explored")
                end
            else
                println("All nodes have been explored")
            end
            break
        end

        global_lower_bound = master_node_list[1].lower_bound
        for current_node in master_node_list
            if current_node.lower_bound < global_lower_bound
                global_lower_bound = current_node.lower_bound
            end
        end

        if num_explored_nodes % 50 == 0
            current_time = now() - start_time
            if (Dates.value(current_time) / 1000 / 60) > cutoff_time
                break
            end
        end
    end

    total_time = now() - start_time

    if output_to_file
        open(output_file_name, "a") do io
            write(io, "An optimal solution has been found!")
        end
    else
        println("An optimal solution has been found!")
    end

    if BPD_backbone
        temp_sol = zeros(original_n)
        for i=1:n
            temp_sol[backbone[i]] = global_x_sol[i]
        end
        global_x_sol = temp_sol
    end

    return global_x_sol,
           global_upper_bound,
           global_lower_bound,
           num_explored_nodes,
           upper_bound_hist,
           lower_bound_hist,
           total_time,
           terminal_nodes

end;
