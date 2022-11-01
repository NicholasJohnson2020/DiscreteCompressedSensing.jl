function functionEvalV2(lambda, sigma, nu, constant)

    val = 0

    numerator = (nu .^2) .* (sigma .+ (2/lambda))
    denominator = (sigma .+ (1/lambda)) .^2

    return constant - sum(numerator ./ denominator)

end

#Line Search 2
function LineSearchV2(A, b, z, epsilon, gamma;
        termination_threshold=1e-2, lambda_init=1)

    (m, n) = size(A)
    z_ones = findall(x -> x>0.5, z)
    local_A = A[:, z_ones]

    local_A = convert(Array{Float64,2}, local_A)
    b = convert(Array{Float64, 1}, b)

    min_x = local_A \ b
    if norm(local_A*min_x-b)^2 > epsilon
        return (false)
    end

    svd_output = svd(local_A' * local_A)
    sigma = svd_output.S
    U = svd_output.U
    nu = U' * local_A' * b
    constant = b'*b - epsilon

    mid_lambda = lambda_init
    upper_lambda = mid_lambda
    lower_lambda = mid_lambda

    current_value = functionEvalV2(lambda_init, sigma, nu, constant)
    if current_value < 0
       while current_value < 0
            upper_lambda = lower_lambda
            lower_lambda = lower_lambda / 2
            current_value = functionEvalV2(lower_lambda, sigma, nu, constant)
            mid_lambda = lower_lambda
        end
    else
        while current_value > 0
            lower_lambda = upper_lambda
            upper_lambda = upper_lambda * 2
            current_value = functionEvalV2(upper_lambda, sigma, nu, constant)
            mid_lambda = upper_lambda
        end
    end

    while abs(current_value) > termination_threshold
        mid_lambda = (lower_lambda + upper_lambda) / 2
        current_value = functionEvalV2(mid_lambda, sigma, nu, constant)
        if current_value > 0
            lower_lambda = mid_lambda
        else
            upper_lambda = mid_lambda
        end
    end

    fit_x = U * Diagonal(1 ./ (sigma .+ 1/mid_lambda)) * nu

    opt_x = zeros(n)
    opt_grad = ones(n)
    for i=1:size(z_ones)[1]
        opt_x[z_ones[i]] = fit_x[i]
        opt_grad[z_ones[i]] -= fit_x[i]^2/gamma
    end

    return (true, size(z_ones)[1] + (fit_x'*fit_x)/gamma, opt_grad, opt_x)

end;

function CuttingPlanesV2(A, b, epsilon, lambda; solver_output=0,
    sparsity_lower_bound=0, lower_bound_obj=nothing, upper_bound_x_sol=nothing)
    (m, n) = size(A)

    miop = direct_model(Gurobi.Optimizer(GUROBI_ENV))
    #miop = direct_model(Gurobi.Optimizer())
    set_optimizer_attribute(miop, "OutputFlag", solver_output)

    # Optimization variables
    @variable(miop, z[1:n], Bin)
    @variable(miop, t >= 0)
    # Objective
    @objective(miop, Min, t)
    # Constraints
    @constraint(miop, sum(z) >= sparsity_lower_bound)

    optimal_x = zeros(n)
    optimal_value = 0
    z0 = ones(n)
    output = LineSearchV2(A, b, z0, epsilon, lambda)
    if output[1]
        p0 = output[2]
        grad_z0 = output[3]
        optimal_x = output[4]
        optimal_value = p0
        @constraint(miop, t >= p0 + dot(grad_z0, z - z0))
    else
        z_zeros = findall(x -> x<=0.5, z0)
        @constraint(miop, sum(z[i] for i in z_zeros) >= 1)
    end

    if upper_bound_x_sol != nothing
        z0 = abs.(upper_bound_x_sol) .> 1e-4
        @constraint(miop, sum(z) <= sum(z0))
        output = LineSearchV2(A, b, z0, epsilon, lambda)
        if output[1]
            p0 = output[2]
            grad_z0 = output[3]
            optimal_x = output[4]
            optimal_value = p0
            @constraint(miop, t >= p0 + dot(grad_z0, z - z0))
        else
            z_zeros = findall(x -> x<=0.5, z0)
            @constraint(miop, sum(z[i] for i in z_zeros) >= 1)
        end
    end

    if lower_bound_obj != nothing
        @constraint(miop, t >= lower_bound_obj)
    end


    cb_calls = Cint[]
    z_hist = []
    function outer_approximation_opt(cb_data, cb_where::Cint)
        push!(cb_calls, cb_where)
        if cb_where != GRB_CB_MIPSOL && cb_where != GRB_CB_MIPNODE
            return
        end
        if cb_where == GRB_CB_MIPNODE
            resultP = Ref{Cint}()
            GRBcbget(cb_data, cb_where, GRB_CB_MIPNODE_STATUS, resultP)
            if resultP[] != GRB_OPTIMAL
                return
            end
        end
        Gurobi.load_callback_variable_primal(cb_data, cb_where)
        z_val = callback_value.(cb_data, z)
        t_val = callback_value(cb_data, t)
        output = LineSearchV2(A, b, z_val, epsilon, lambda)
        append!(z_hist, [z_val])
        if output[1]
            obj = output[2]
            grad_z = output[3]
            optimal_x = output[4]
            optimal_value = obj
            offset = sum(grad_z .* z_val)
            if t_val < obj
                con = @build_constraint(t >= obj + sum(grad_z[j] * z[j] for j=1:n) - offset)
                MOI.submit(miop, MOI.LazyConstraint(cb_data), con)
            end
        else
            z_zeros = findall(x -> x<=0.5, z_val)
            con = @build_constraint(sum(z[i] for i in z_zeros) >= 1)
            MOI.submit(miop, MOI.LazyConstraint(cb_data), con)
        end
    end
    MOI.set(miop, MOI.RawParameter("LazyConstraints"), 1)
    MOI.set(miop, Gurobi.CallbackFunction(), outer_approximation_opt)

    optimize!(miop)
    z_opt = JuMP.value.(z)

    return optimal_x, z_opt, optimal_value, size(cb_calls)[1], z_hist
end;
