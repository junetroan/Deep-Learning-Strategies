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
state = 2
U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))
p, st = Lux.setup(rng1, U)
K = Float32[0.214566, 0.928355]


#Definition of the model 
function predictor!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    yt = y_zoh(t)
    e = yt .- û[1]
    du[1:end] =  û[1:end] .+ abs.(p.K) .* e
end

params = ComponentVector{Float32}(vector_field_model = p, K = K)
#ps = getdata(params)
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

#p0 = [0.7, 1.0] 
# Optimization to find the best hyperparameters
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> predloss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters = 500, verbose = false)

# Predictions
y_pred = prediction(res_pred.u)

plot(y_pred[1,:])


