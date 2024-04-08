# From ANODE-MS repository - there wrongfully named NODE-GW Model

# SciML Tools
using Pkg
using DifferentialEquations
using SciMLSensitivity 
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Distributions
using DiffEqFlux: NeuralODE, AdamW, swish

# Standard Libraries
using LinearAlgebra, Statistics

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, StableRNGs
plotly()

# Set a random seed for reproducible behaviour
rng = StableRNG(1111)

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Define the experimental parameter
tspan = (0.0f0, 10.0f0) # Original (0.0f0, 10.0f0)
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
println("solving ODE")
@time solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25)
num_elements = 5

# Add noise in terms of the mean
X = Array(solution)
t = solution.t

x̄ = mean(X, dims = 2)
noise_magnitude = 10e-3
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])

#------ Building neural ODE
# Simple NN to predict lotka-volterra dynamics
U = Lux.Chain(Lux.Dense(2, 30, tanh),
              Lux.Dense(30, 2))
# Get the initial parameters and state variables of the model
p_lv, st_lv = Lux.setup(rng, U)

# Define the hybrid model
function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p.vector_field_model, st_lv)[1] # Network prediction
    du[1] = û[1]
    du[2] = û[2]
end

# Closure with the known parameter
nn_dynamics!(du, u, p_lv, t) = ude_dynamics!(du, u, p_lv, t, p_)

# Construct ODE Problem
augmented_u0 = vcat(u0[1], randn(rng, Float32, 1))
parameters = ComponentVector(vector_field_model = p_lv)
prob_nn = ODEProblem(nn_dynamics!, augmented_u0, tspan, parameters, saveat = 0.25)

losses = Float32[]

function predict(θ, t)
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
    new_problem = remake(prob_nn; p = θ, tspan = (0.0f0, t))
    Array(solve(new_problem, AutoTsit5(Rosenbrock23()), abstol = 1e-6, reltol = 1e-6, sensealg = sensealg))  
end

@time X_ = predict(parameters, t[end])

function loss(θ, y, t)
    X̂ = predict(θ, t)
    prediction_error = mean(abs2, y .- X̂[1,:])
    prediction_error
end

@time l0 = loss(parameters, Xₙ[1, :], t[end])

#@time Zygote.gradient(p -> loss(p, Xₙ[1, :], t[end]), parameters)

function train_one_round(θ, t, y, opt, maxiters; kwargs...)
    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((θ,p) -> loss(θ,y,t), adtype)
    optprob = OptimizationProblem(optf, θ)
    res = Optimization.solve(optprob, opt, maxiters=maxiters; kwargs...)
    res.u
end

t_train = t
y_train = Xₙ[1,:]
obs_grid = num_elements:num_elements:length(t_train)
maxiters = 5000
lr = 5e-4
Xₙ[1, 1:obs_grid[1]]
t[obs_grid[1]]
s = train_one_round(parameters, t[obs_grid[1]], Xₙ[1, 1:obs_grid[1]], ADAM(), 1000; rng)

function train(t, y, obs_grid, maxiters, lr, θ = nothing; kwargs...)
    log_results(θs, losses) = (θ, loss) -> begin
        push!(θs, copy(θ))
        push!(losses, loss)
        false
    end

    θs, losses = ComponentArray[], Float32[]
    
    for i in obs_grid
        θ_new = parameters
        if θ === nothing θ = θ_new end

        t_data = t[i]
        y_data = y[1:i]

        # Testing 
        println("t_data = ", t_data)
        println("y_data = ", y_data)
        println("length of y_data = ", length(y_data))
        
        # Training model with current data
        θ = train_one_round(θ, t_data, y_data, ADAM(), maxiters, callback = log_results(θs, losses), kwargs...)

    end
    θs, losses
end

@time θs, losses = train(t_train, y_train, obs_grid, maxiters, lr);

res_msu = θs[end]

full_traj = predict(res_msu, t_train[end])


###################################################################################################
# Testing 
prob_new = ODEProblem(lotka!, u0, (0.0f0, 40.0f0), p_)
@time solution_new = solve(prob_new, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)
prob_nn_updated = remake(prob_nn, p = res_msu, u0 = u0, tspan = (0.0f0, 40.0f0))
prediction_new = Array(solve(prob_nn_updated, AutoTsit5(Rosenbrock23()),  abstol = 1f-6, reltol = 1f-6,
saveat = 0.25f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))

updated_obsgrid = 0.0:0.25:40.0

t1 = t
t3 = updated_obsgrid[41:123] |> collect

gr()
function plot_results(t, real, pred, pred_new)
    plot(t1, pred[1,:], label = "Training Prediction", title="Training and Test Predictions of ANODE-GW Model", xlabel="Time", ylabel="Population")
    plot!(t3, pred_new[1,41:123], label = "Test Prediction")
    scatter!(t1, real[1,:], label = "Training Data")
    scatter!(t3, solution_new[1,41:123], label = "Test Data")
    vline!([t1[41]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("Results/gw-test-results.png")
end

plot_results(t, Xₙ, full_traj, prediction_new)

actual_loss = solution_new[1,:] - prediction_new[1,:]
total_loss = abs(sum(actual_loss))
total_loss2 = abs2(sum(actual_loss))
MSE = total_loss2 / length(solution_new[1,:])

function plot_scatter(predicted, real)
    plot([minimum(predicted[1,:]), maximum(predicted[1,:])], [minimum(predicted[1,:]), maximum(predicted[1,:])], label="y=x", color=:red, title="Parity plot for the ANODE-GW model")
    scatter!(real[1,:], predicted[1,:], label = "Residuals", xlabel= "Actual values", ylabel="Predicted values", color=RGB(165/225,113/225,220/225))
    #savefig("Results/gw-parity.png")
end

plot_scatter(prediction_new, solution_new)

function plot_results(t, real, pred)
    plot(t, pred[1,:], label = "Prediction")
    scatter!(t, real[1,:], label = "Data")
    #savefig("Results/gw-results.png")
end

####################################################################################################
#=
@time X̂_ = predict(θs[end], t_train[end])

plot(X̂_[1,:], label = "Predicted")

plot(full_traj2[1, :])
scatter!(Xₙ[1, :])
actual_loss = Xₙ[1,:] - full_traj2[1,:]
total_loss = abs(sum(actual_loss))
N = length(y_train)
MSE = (1/N)*sum((Xₙ[1,:] .- full_traj2[1,:]).^2)

function plot_scatter(predicted, real)
    scatter(real[1,:], predicted[1,:], label = "Residuals", xlabel= "Actual values", ylabel="Predicted values")
    plot!([minimum(predicted[1,:]), maximum(predicted[1,:])], [minimum(predicted[1,:]), maximum(predicted[1,:])], label="y=x")
    #savefig("Results/gw-residuals.png")
end

plot_scatter(full_traj2, Xₙ)

function predict_final(θ)
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
    prob_nn_updated = remake(prob_nn, p=θ, u0=X̂_[:,1]) # no longer updates u0 nn
    X̂ = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)), abstol = 1f-6, reltol = 1f-6, saveat =0.25f0,sensealg = sensealg)) 
    X̂
end
 
function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = mean(abs2, Xₙ[1,:] - X̂[1,:])
    prediction_error
end

#final_traj = predict_final(θs[end])

optf_final = Optimization.OptimizationFunction((x,p) -> final_loss(x), adtype)
optprob_final = Optimization.OptimizationProblem(optf_final, res_ms.u)
res_final = Optimization.solve(optprob_final, BFGS(initial_stepnorm = 0.01), callback=callback, maxiters = 1000, allow_f_increases = true)

full_traj2 = predict_final(res_final.u)
actual_loss = Xₙ[1,:] - full_traj2[1,:]
total_loss = abs(sum(actual_loss))

gr()

function plot_results(t, real, pred)
    plot(t, pred[1,:], label = "Prediction")
    scatter!(t, real[1,:], label = "Data")
    #savefig("Results/gw-results.png")
end

plot_results(t, Xₙ, final_traj)


actual_loss = Xₙ[1,:] - final_traj[1,:]
total_loss = abs(sum(actual_loss))

function plot_residuals(t,residuals)
    plot(t, residuals, label = "Residuals", xlabel= "Time", ylabel="Residuals")
    #savefig("Results/gw-residuals-time.png")
end

plot_residuals(t,actual_loss)


function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = mean(abs2, Xₙ[1,:] - X̂[1,:])
    prediction_error
end

pred_error = final_loss(θs[end])

function final_trained_model(θ)
    callback = function (p, l)
        push!(losses, l)
        if length(losses) % 50 == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        return false
    end
    
    adtype = Optimization.AutoZygote()
    optf_final = OptimizationFunction((θ,p) -> final_loss(θ), adtype)
    optprob_final = OptimizationProblem(optf_final, θ)
    #res_final = solve(optprob_final, BFGS(initial_stepnorm = 0.01), callback = callback, maxiters=1000, allow_f_increases = true)
    res_final = solve(optprob_final, ADAM(), callback = callback, maxiters=1000, allow_f_increases = true)
    res_final.u
end

res_fin = final_trained_model(θs[end])

p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob_new = ODEProblem(lotka!, u0, (0.0f0, 40.0f0), p_)
@time solution_new = solve(prob_new, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)
prob_nn_updated = remake(prob_nn, p = res_fin, u0 = X̂_[:,1], tspan = (0.0f0, 40.0f0))
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)),  abstol = 1f-6, reltol = 1f-6,
saveat = 0.25f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))

plot(0.0:0.25:40.0 |> collect, prediction_new[1, :])
scatter!(0.0:0.25:40.0 |> collect, solution_new[1, :])
=#