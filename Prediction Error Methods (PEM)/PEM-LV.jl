using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using CSV, Tables, DataFrames
using DataInterpolations
using OptimizationPolyalgorithms
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

# Interpolation of given data
y_zho = ConstantInterpolation(y, tsteps)

# Definition of neural network
state = 2
U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))
p_init, st = Lux.setup(rng1, U)
params = ComponentVector{Float32}(ANN = p_init, u0_states = 0.5*randn(rng2, Float32, state - 1))

#Definition of the model 
function predictor(du,u,p,t)
    K, p_nn, st = p # Gain and neural network parameters
    û = U(u, p_nn.ANN, st)[1] # Network prediction
    yt = y_zho
    e = yt - û[1]
    du[1:end] = û[1:end] + K*e
end

predprob = ODEProblem(predictor, [u0[1]; params.u0_states], tspan, params, saveat=tsteps, p = (0.5, p_init, st))

# Predict function
function prediction(p)
    p_full = (p..., y_zho)
    _prob = remake(predprob, u0 = u0, p = p_full)
    solve(_prob, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)
end

# Loss function
function predloss(p)
    yh = prediction(p)
    e2 = mean(abs2, y .- yh)
    return e2
end

# Optimization to find the best hyperparameters
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> predloss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(p0))
res_pred = Optimization.solve(optprob,PolyOpt(), maxiters=5000)

# Simulation of the model with the best hyperparameters
function simulate(p)
    _prob = remake(prob,p=p)
    solve(_prob, Tsit5(), saveat=tsteps, abstol = 1e-8, reltol = 1e-6)[1,:]
end

# Predictions
y_pred = simulate(res_pred.u)

# Loss function of the predictions vs. the given data
function simloss(p)
    yh = simulate(p)
    e2 = yh
    e2 .= abs2.(y .- yh)
    return mean(e2)
end



