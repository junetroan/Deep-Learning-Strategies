# Created on Monday 8th of April by collaboration of Vinicius Santana Viena and June Mari Berge Trøan


# GENERATING DATA TO TEST
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots
using DiffEqFlux
using Plots
using Distributions

gr()
rng = StableRNG(1111)

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

tspan = (0.0, 10.0)
u0 = 5.0 * rand(rng, Float64, 2)
p_ = [1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
println("solving ODE")
@time solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25)
X = Array(solution)
x = X[1,:]
#x2 = zeros(length(x))
#X = [x x2]'
# SPECIFICATIONS
group_size = 5
state = 2
groupsize = 5
predsize = 5
tsteps = 0.25

u0 = Float64(x[first(1:5)])
u0_init = [[x[first(rg)]; fill(mean(x[rg]), state - 1)] for rg in ranges] 
u0_init = mapreduce(permutedims, vcat, u0_init)

# NEURAL NETWORK
U = Lux.Chain(Lux.Dense(state, 30, tanh),
              Lux.Dense(30, state))
              
p, st = Lux.setup(rng, U)
params = ComponentVector{Float64}(θ = p, u0_init = u0)
neuralode = NeuralODE(U, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
prob_node = ODEProblem((u,p,t) -> U(u, p, st)[1][1:end], u0, tspan, params)

datasize = size(x, 1)
tspan = (0.0, 10.0) # Original (0.0f0, 10.0f0
tsteps = range(tspan[1], tspan[2]; length = datasize)
#solve(prob_node, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = 0.25f0)

##############################################################################################################################################################
# Testing to reveal variable types and variable content

datasize
groupsize
x
state

##############################################################################################################################################################

function group_ranges(datasize::Integer, groupsize::Integer)
    2 <= groupsize <= datasize || throw(DomainError(groupsize,
        "datasize must be positive and groupsize must to be within [2, datasize]"))
    return [i:min(datasize, i + groupsize - 1) for i in 1:(groupsize - 1):(datasize - 1)]
end

ranges = group_ranges(datasize, group_size)


function multiple_shoot_mod(p, ode_data, tsteps, prob::ODEProblem, loss_function::F,
    continuity_loss::C, solver::SciMLBase.AbstractODEAlgorithm, group_size::Integer;
    continuity_term::Real = 100, kwargs...) where {F, C}

    datasize = size(ode_data, 1)

    if group_size < 2 || group_size > datasize
        throw(DomainError(group_size, "group_size can't be < 2 or > number of data points"))
    end

    ranges = group_ranges(datasize, group_size)

    sols = [solve(remake(prob_node; θ = p.θ, tspan = (tsteps[first(rg)], tsteps[last(rg)]),
        u0 = p.u0_init[index, :]),
        solver, saveat = tsteps[rg], sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)))
        for (index, rg) in enumerate(ranges)]

    group_predictions = Array.(sols)

    loss = 0

    for (i, rg) in enumerate(ranges)
        u = x[rg] # TODO: make it generic for observed states > 1
        û = group_predictions[i][1, :]
        loss += loss_function(u, û)

        if i > 1
            # Ensure continuity between last state in previous prediction
            # and current initial condition in ode_data
            loss += continuity_term *
                    continuity_loss(group_predictions[i - 1][:, end], group_predictions[i][:, 1])
        end
    end

    return loss, group_predictions
end

#Testing
# Calculate multiple shooting loss
loss_function(data, pred) = sum(abs2, data - pred)
continuity_loss(uᵢ₊₁, uᵢ) = sum(abs2, uᵢ₊₁ - uᵢ)
predict_single_shooting(p) = Array(first(neuralode(u0_init[1,:],p,st)))
tester_pred_ss = predict_single_shooting(params.θ)

function loss_single_shooting(p)
    pred = predict_single_shooting(p)
    l = loss_function(x, pred[1,:])
    return l, pred
end

ls, ps = loss_single_shooting(params.θ)

continuity_term = 10.0
#pars = ComponentVector{Float64}(θ = p, u0_init = u0_init)
ls, ps = multiple_shoot_mod(params, x, tsteps, prob_node, loss_function, continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size; continuity_term)

#lossses, preds = predict_single_shooting(params.p)

# Modified multiple_shoot method 
multiple_shoot_mod(pars, x, tsteps, prob_node, loss_function,
    continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size;
    continuity_term)

function loss_multiple_shoot(p)
    return multiple_shoot_mod(p, x, tsteps, prob_node, loss_function,
        continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size;
        continuity_term)[1]
end

test_multiple_shoot = loss_multiple_shoot(pars)
#test_multiple_shoot[1]

loss_single_shooting(pars.θ)[1]

losses = Float64[]

callback = function (p, l)
    push!(losses, loss_multiple_shoot(p))
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")

    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shoot(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pars)
@time res_ms = Optimization.solve(optprob, ADAM(),  maxiters = 5000, callback = callback)

loss_ms, _ = loss_single_shooting(res_ms.u.θ)
preds = predict_single_shooting(res_ms.u.θ)

plot(preds[1, :])
scatter!(x)

# Plotting parity
scatter(x, preds[1,:], label = "Data", color = :purple)
plot!(x, x, label = "Parity", color = :orange) # Add this line


#####################################################################################################################################################################
# Test with BFGS
# Does not work with current implementation

u0_inits = res_ms.u.u0_init
u0_inits_all = Matrix{Float64}(u0_inits)
#probnnup = remake(prob_node, θ = res_ms.u.θ, u0_init = u0_inits_all)
probnnup = remake(prob_node, θ = res_ms.u.θ, u0_init = u0_inits_all)
sols = Array(solve(probnnup, AutoVern7(KenCarp4(autodiff = true)), abstol = 1e-6, reltol = 1e-6, saveat = 0.25, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))


function predict_final(θ)
    u0_all = θ.u0_init
    u0_all = Matrix{Float64}(u0_all)
    prob_nn_updated = remake(prob_node, θ = θ, u0_init = u0_all)
    X̂ = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)), abstol = 1e-6, reltol = 1e-6, saveat = 0.25, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
    return X̂
end

predict_final(res_ms.u)

function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = sum(abs2, x .- X̂[1, :])
    return prediction_error
end

optf_final = Optimization.OptimizationFunction((x,p) -> final_loss(x), adtype)
optprob_final = Optimization.OptimizationProblem(optf_final, res_ms.u)
@time res_final = Optimization.solve(optprob_final, BFGS(initial_stepnorm = 0.01), callback=callback, maxiters = 1000, allow_f_increases = true)

full_traj2 = predict_final(res_final.u)
actual_loss = Xₙ[1,:] - full_traj2[1,:]
total_loss = abs(sum(actual_loss))

final_full_trajectory_loss = final_loss(res_final.u)

#####################################################################################################################################################################
# Testing 
u0 = vcat(res_ms.u.u0_init[1,:])
p_ = [1.3, 0.9, 0.8, 1.8]
prob_new = ODEProblem(lotka!, u0, (0.0, 40.0), p_)
@time solution_new = solve(prob_new, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25)
tsteps = 0.0:0.25:40.0

x_test = solution_new[1,:]

prob_nn_updated = remake(prob_node, u0 = u0, tspan = (0.0, 40.0), p = res_ms.u.θ)
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)), abstol = 1e-6, reltol = 1e-6, saveat = 0.25, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
t_train  = 0.0:0.25:10.0 |> collect
t1 = t_train |> collect
t3 = tsteps[41:123] |> collect

gr()

function plot_results(t, real, pred, pred_new)
    plot(t1, pred[1,:], label = "Training Prediction", title="Training and Test Predictions of MNODE-MS Model", xlabel = "Time", ylabel = "Population")
    plot!(t3, pred_new[1,41:123], label = "Test Prediction")
    scatter!(t1, real, label = "Training Data")
    scatter!(t3, solution_new[1,41:123], label = "Test Data")
    vline!([t3[1]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("Results/LV/MNODE-MS LV Training and Testing.png")
end

plot_results(t1, x, preds, prediction_new)

actual_loss = solution_new[1,:] - prediction_new[1,:]
total_loss = abs(sum(actual_loss))