function perspectiveFormulationDual(A, b, z, epsilon, gamma; solver_output=0, solver="Gurobi", M=100)

    @assert solver in ["Gurobi", "SCS"]

    (m, n) = size(A)

    if solver == "Gurobi"
        model = Model(with_optimizer(Gurobi.Optimizer, GUROBI_ENV))
        set_optimizer_attribute(model, "OutputFlag", solver_output)
    else
        model = Model(SCS.Optimizer)
        set_optimizer_attribute(model, "verbose", solver_output)
    end

    @variable(model, mu_1[i=1:m])
    @variable(model, mu_2[i=1:n])
    @variable(model, nu[i=1:n])
    @variable(model, lamb[i=1:n] >= 0)
    @variable(model, sigma >= 0)

    @constraint(model, nu + A'*mu_1 + mu_2 .== 0)
    @constraint(model, [i=1:n], nu[i] <= lamb[i])
    @constraint(model, [i=1:n], nu[i] >= -lamb[i])
    @constraint(model, [sigma; mu_1] in SecondOrderCone())
    @constraint(model, sum(mu_2[i]^2 for i=1:n) <= 1/gamma)

    @objective(model, Max, -M * sum(lamb[i]*z[i] for i=1:n) - sigma * epsilon^0.5 - mu_1'*b)

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        obj_value = objective_value(model)
        z_gradient = - M * value.(lamb)
        return termination_status(model), (obj_value^2 + sum(z), 2 * obj_value * z_gradient .+ 1)
    else
        return termination_status(model), nothing
    end

end;

function perspectiveFormulationSub(A, b, z, epsilon, gamma; solver_output=0, solver="Gurobi", M=100)

    @assert solver in ["Gurobi", "SCS"]

    (m, n) = size(A)

    if solver == "Gurobi"
        model = Model(with_optimizer(Gurobi.Optimizer, GUROBI_ENV))
        set_optimizer_attribute(model, "OutputFlag", solver_output)
    else
        model = Model(SCS.Optimizer)
        set_optimizer_attribute(model, "verbose", solver_output)
    end

    @variable(model, x[i=1:n])
    @variable(model, residual[i=1:m])
    @variable(model, t >= 0)

    @constraint(model, residual .== A * x .- b)
    @constraint(model, sum(residual[i]^2 for i=1:m) <= epsilon)
    @constraint(model, sum(x[i]^2 for i=1:n) <= t)

    @constraint(model, lamb_plus[i=1:n], x[i] <= z[i] * M)
    @constraint(model, lamb_min[i=1:n], x[i] >= -z[i] * M)

    @objective(model, Min, t/gamma)

    optimize!(model)

    obj_value = objective_value(model)
    opt_x = value.(x)

    return opt_x, obj_value

end;

function CuttingPlanes(A, b, epsilon, lambda; solver_output=0, M=100, lower_bound=1, upper_bound=nothing)
    (m, n) = size(A)

    miop = direct_model(Gurobi.Optimizer(GUROBI_ENV))
    set_optimizer_attribute(miop, "OutputFlag", solver_output)

    # Optimization variables
    @variable(miop, z[1:n], Bin)
    @variable(miop, t >= 0)
    # Objective
    @objective(miop, Min, t)
    # Constraints
    @constraint(miop, sum(z) >= lower_bound)
    if upper_bound != nothing
        @constraint(miop, sum(z) <= upper_bound)
    end
    z0 = rand(Bernoulli(0.5), n)
    status, output = perspectiveFormulationDual(A, b, z0, epsilon, lambda, M=M)
    if status == MOI.OPTIMAL
        (p0, grad_z0) = output
        @constraint(miop, t >= p0 + dot(grad_z0, z - z0))
    else
        #z_nonzeros = findall(x -> x>0.5, z0)
        z_zeros = findall(x -> x<=0.5, z0)
        #@constraint(miop, sum(z[i] for i in z_nonzeros) + sum(1-z[i] for i in z_zeros) <= n-1)
        @constraint(miop, sum(z[i] for i in z_zeros) >= 1)
    end
    cb_calls = Cint[]
    function outer_approximation(cb_data, cb_where::Cint)
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
        status, output = perspectiveFormulationDual(A, b, z_val, epsilon, lambda, M=M)
        if status == MOI.OPTIMAL
            (obj, grad_z) = output
            offset = sum(grad_z .* z_val)
            if t_val < obj
                con = @build_constraint(t >= obj + sum(grad_z[j] * z[j] for j=1:n) - offset)
                MOI.submit(miop, MOI.LazyConstraint(cb_data), con)
            end
        else
            #z_nonzeros = findall(x -> x>0.5, z_val)
            z_zeros = findall(x -> x<=0.5, z_val)
            #con = @build_constraint(sum(z[i] for i in z_nonzeros) + sum(1-z[i] for i in z_zeros) <= n-1)
            con = @build_constraint(sum(z[i] for i in z_zeros) >= 1)
            MOI.submit(miop, MOI.LazyConstraint(cb_data), con)
        end
    end
    MOI.set(miop, MOI.RawParameter("LazyConstraints"), 1)
    MOI.set(miop, Gurobi.CallbackFunction(), outer_approximation)

    optimize!(miop)
    z_opt = JuMP.value.(z)

    opt_sol, opt_val = perspectiveFormulationSub(A, b, z_opt, epsilon, lambda,
                                                 solver_output=solver_output, M=100)

    return opt_sol, z_opt, opt_val + sum(z_opt)
end;
