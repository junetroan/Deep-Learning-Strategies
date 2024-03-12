using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using CSV, Tables, DataFrames
using DataInterpolations
using OrdinaryDiffEq
using OptimizationPolyalgorithms
using DiffEqFlux
using Plots
plotly()


# Producing data
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

rng = StableRNG(1111)
rng1 = StableRNG(1003)
rng2 = StableRNG(1004)
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
tspan = (0.0f0, 10.0f0)
prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)
X = Array(solution)
t = solution.t
tspan = (t[1], t[end])
tsteps = range(tspan[1], tspan[2], length = length(X[1,:]))


# Adding noise
x̄ = mean(X, dims = 2)
noise_magnitude = 10f-3
noise_magnitude2 = 62.3f-2
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))
y = Xₙ[1,:]

# Definition of neural network
state = 4
U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))
p, st = Lux.setup(rng1, U)
K = Float32[0.214566,0.028355, 0.01111, 0.022111]

y_zoh = ConstantInterpolation(y, tsteps)
u0 = [0.0f0 , 0.0f0, 0.0f0, 0.0f0]

#Definition of the model 
function predictor!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    yt = y_zoh(t)
    e = yt .- û[1]
    du[1:end] =  û[1:end] .+ abs.(p.K) .* e
end

params = ComponentVector{Float32}(vector_field_model = p, K = K)
prob_nn = ODEProblem(predictor!, u0 , tspan, params, saveat=tsteps)
soln_nn = Array(solve(prob_nn, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0))

# Predict function
function prediction(p)
    _prob = remake(prob_nn, u0 = u0, p = p)
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
    Array(solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0, sensealg = sensealg))
end

# Loss function
function predloss(p)
    yh = prediction(p)
    e2 = mean(abs2, y .- yh[1,:])
    return e2
end

losses = Float32[]
callback = function (θ, l)
    push!(losses, predloss(θ))

    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end
 
# Optimization to find the best hyperparameters
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> predloss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters = 5000, verbose = false, callback  = callback)

# Predictions
y_pred = prediction(res_ms.u)
plot(y_pred[1,:])

# Testing
## Generating data
tspan_test = (0.0f0, 40.0f0)
prob_test = ODEProblem(lotka!, u0, tspan_test, p_)
solution_test = solve(prob_test, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)
X_test = Array(solution_test)
t_test = solution_test.t
tspan_test = (t_test[1], t_test[end])
tsteps_test = range(tspan_test[1], tspan_test[2], length = length(X_test[1,:]))
y_test = X_test[1,:]
y_zoh2 = ConstantInterpolation(y_test, tsteps_test)

plot(y_test)
plot!(y_zoh2)

function simulator!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    du[1:end] =  û[1:end]
end

u0 = [0.0f0, 0.0f0, 0.0f0, 0.0f0]
params_test = ComponentVector{Float32}(vector_field_model = p)
prob_test = ODEProblem(simulator!, u0 , tspan_test, params_test, saveat=tsteps_test)
prob = remake(prob_test, p = res_ms.u, tspan = tspan_test)
soln_nn = Array(solve(prob, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0))

plot(t_test, soln_nn[1,:])
plot!(t_test, y_test)

function plot_results(t, real, real_new,  pred, pred_new)
    plot(t, pred[1,:],label = "Training Prediction", title = "Training and Test Predicitons of PEM Model", xlabel = "Time", ylabel = "Population")
    plot!(t, pred_new[1,pred[end]:end], label = "Test Prediction")
    scatter!(t, real, label = "Training Data")
    scatter!(t, real_new, label = "Test Data")
    vline!(t[pred[end]], label = "Training/Test Split", color = :black)    
end