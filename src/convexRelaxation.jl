function perspectiveRelaxation(A, b, epsilon, lambda;
    solver_output=0, solver="Gurobi", round_solution=true)

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
    @variable(model, z[i=1:n] >= 0)
    @variable(model, theta[i=1:n] >= 0)
    @variable(model, residual[i=1:m])

    @constraint(model, residual .== A * x .- b)
    @constraint(model, sum(residual[i]^2 for i=1:m) <= epsilon)
    #@constraint(model, [epsilon^0.5; A*x.-b] in SecondOrderCone())

    @constraint(model, [i=1:n], z[i] * theta[i] >= x[i]^2)

    @constraint(model, [i=1:n], z[i] <= 1)

    @objective(model, Min, sum(z[i] + theta[i] / lambda for i=1:n))

    optimize!(model)

    obj_value = objective_value(model)
    opt_x = value.(x)
    opt_z = value.(z)

    if round_solution
        rounding_start_z = now()
        rounded_xz, num_support_z = roundSolution(opt_z, A, b, epsilon)
        rounding_end_z = now()
        rounding_time_z = rounding_end_z - rounding_start_z

        rounding_start_x = now()
        rounded_xx, num_support_x = roundSolution(opt_x, A, b, epsilon)
        rounding_end_x = now()
        rounding_time_x = rounding_end_x - rounding_start_x

        return (num_support_z,
                rounded_xz,
                num_support_x,
                rounded_xx,
                opt_x, opt_z, obj_value,
                rounding_time_z,
                rounding_time_x)
    else
        return opt_x, opt_z, obj_value
    end

end;


function perspectiveFormulation(A, b, epsilon, lambda; norm_function="L2",
    solver_output=0, solver="Gurobi", BPD_backbone=false)

    @assert solver in ["Gurobi", "SCS"]
    @assert norm_function in ["L2", "L1"]

    (m, n) = size(A)
    original_n = n
    backbone = []

    if BPD_backbone

        _, opt_x = basisPursuitDenoising(A, b, epsilon,
                                         norm_function=norm_function,
                                         round_solution=false)

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
    end

    if solver == "Gurobi"
        model = Model(with_optimizer(Gurobi.Optimizer, GUROBI_ENV))
        set_optimizer_attribute(model, "OutputFlag", solver_output)
    else
        model = Model(SCS.Optimizer)
        set_optimizer_attribute(model, "verbose", solver_output)
    end

    @variable(model, x[i=1:n])
    @variable(model, z[i=1:n], Bin)
    @variable(model, theta[i=1:n] >= 0)
    @variable(model, abs_residual[i=1:m])

    @constraint(model, abs_residual .>= A * x .- b)
    @constraint(model, abs_residual .>= -A * x .+ b)

    if norm_function == "L2"
        @constraint(model, sum(abs_residual[i]^2 for i=1:m) <= epsilon)
    else
        @constraint(model, sum(abs_residual[i] for i=1:m) <= epsilon)
    end

    @constraint(model, [i=1:n], z[i] * theta[i] >= x[i]^2)

    @objective(model, Min, sum(z[i] + theta[i] / lambda for i=1:n))

    optimize!(model)

    println(termination_status(model))
    obj_value = objective_value(model)
    opt_x = value.(x)
    opt_z = value.(z)

    if BPD_backbone
        temp_x = zeros(original_n)
        temp_z = zeros(original_n)
        for i=1:n
            temp_x[backbone[i]] = opt_x[i]
            temp_z[backbone[i]] = opt_z[i]
        end
        opt_x = temp_x
        opt_z = temp_z
    end

    return opt_x, opt_z, obj_value

end;
