using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using CSV, Tables, DataFrames
gr()

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Experimental parameters
state = 4 # number of state variables in the neural network

tspan = (0.0f0, 10.0f0)
rng = StableRNG(1111)
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)


# Add noise in terms of the mean
X = Array(solution)
t = solution.t
x̄ = mean(X, dims = 2)
noise_magnitude = 10f-3
noise_magnitude2 = 62.3f-2
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))


# Simple NN to predict lotka-volterra dynamics
rng1 = StableRNG(1003)
U = Lux.Chain(Lux.Dense(state, 30, tanh),
Lux.Dense(30, state))
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng1, U)


# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    û = U(u, p.vector_field_model, st)[1] # Network prediction
    du[1:end] = û[1:end]
end

# Construct ODE Problem
augmented_u0 = vcat(u0[1], zeros(Float32, state - 1))
params = ComponentVector{Float32}(vector_field_model = p)
prob_nn = ODEProblem(ude_dynamics!, augmented_u0, tspan, params, saveat = 0.25f0)

sol = solve(prob_nn, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)

plot(sol)