function functionEvalV3(lambda, sigma, nu, constant)

    val = 0

    nu_squared = nu.^2
    sigma_squared = sigma .^ 2
    factor = sigma_squared ./ (sigma_squared .+ (1/lambda))
    val = nu_squared .* factor .* (factor .- 2)

    return sum(val) + constant

end

function derivativeEvalV3(lambda, U, sigma, V, b, A, nu)

    sigma_squared = sigma .^ 2
    vector_1 = U * Diagonal(sigma_squared ./ (sigma_squared .+ 1/lambda)) * nu - b
    vector_2 = Diagonal(sigma .* lambda ./ (sigma_squared .+ 1/lambda)) * nu
    mat_1 = 2 * vector_1 * vector_2' * V'
    df_dz = sum(mat_1 .* A, dims=1)

    return df_dz

end

function LineSearchV3(A, b, z, epsilon, gamma;
        termination_threshold=1e-2, lambda_init=1)

    (m, n) = size(A)
    z_ones = findall(x -> x>0.5, z)
    z_zeros = findall(x -> x<0.5, z)

    A = convert(Array{Float64,2}, A)
    AZ = copy(A)
    for i in z_zeros
        AZ[:, i] = zeros(m)
    end

    b = convert(Array{Float64, 1}, b)

    min_x = AZ \ b
    if norm(AZ*min_x-b)^2 > epsilon
        return (false)
    end

    svd_output = svd(AZ)
    sigma = svd_output.S
    U = svd_output.U
    V = svd_output.Vt'
    nu = U'*b
    constant = b'*b - epsilon

    mid_lambda = lambda_init
    upper_lambda = mid_lambda
    lower_lambda = mid_lambda

    current_value = functionEvalV3(lambda_init, sigma, nu, constant)
    if current_value < 0
       while current_value < 0
            upper_lambda = lower_lambda
            lower_lambda = lower_lambda / 2
            current_value = functionEvalV3(lower_lambda, sigma, nu, constant)
            mid_lambda = lower_lambda
        end
    else
        while current_value > 0
            lower_lambda = upper_lambda
            upper_lambda = upper_lambda * 2
            current_value = functionEvalV3(upper_lambda, sigma, nu, constant)
            mid_lambda = upper_lambda
        end
    end

    while abs(current_value) > termination_threshold
        mid_lambda = (lower_lambda + upper_lambda) / 2
        current_value = functionEvalV3(mid_lambda, sigma, nu, constant)
        if current_value > 0
            lower_lambda = mid_lambda
        else
            upper_lambda = mid_lambda
        end
    end

    fit_grad = derivativeEvalV3(mid_lambda, U, sigma, V, b, A, nu) / gamma
    fit_x = V*Diagonal(sigma ./ (sigma.^2 .+ 1/mid_lambda))*nu

    return (true, size(z_ones)[1] + norm(fit_x)^2/gamma, vec(fit_grad) .+ 1, fit_x, mid_lambda)

end;

function CuttingPlanesV3(A, b, epsilon, lambda; solver_output=0,
    lower_bound_obj=0, upper_bound_x_sol=nothing)
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
    @constraint(miop, sum(z) >= 1)

    optimal_x = zeros(n)
    optimal_value = 0
    z0 = ones(n)
    output = LineSearchV3(A, b, z0, epsilon, lambda)
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
        output = LineSearchV3(A, b, z0, epsilon, lambda)
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
    function outer_approximation_opt(cb_data, cb_where::Cint)
        push!(cb_calls, cb_where)
        num_cuts = size(cb_calls)[1]
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
        output = LineSearchV3(A, b, z_val, epsilon, lambda)
        num_cuts = size(cb_calls)[1]
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

    return optimal_x, z_opt, optimal_value, size(cb_calls)[1]
end;
