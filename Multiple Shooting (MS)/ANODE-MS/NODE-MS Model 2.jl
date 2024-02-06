# Test againt Julias multiple shooting
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Distributions
using ComponentArrays, Lux, DiffEqFlux, Optimization, OptimizationPolyalgorithms, DifferentialEquations, Plots
using DiffEqFlux: group_ranges
using OptimizationOptimisers
using StableRNGs

state = 2
group_size = 5
continuity_term = 10.0f0

rng = StableRNG(1111)
tspan = (0.0f0, 10.0f0) # Original (0.0f0, 10.0f0)
tsteps = 0.25f0
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
datasize = 41
tsteps = range(tspan[1], tspan[2], length = datasize)

# Get the data
function trueODEfunc(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan, p_)
solution = solve(prob_trueode, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = tsteps)

X = Array(solution)
t = solution.t
x̄ = mean(X, dims = 2)
noise_magnitude = 10f-3
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

# Define the Neural Network
nn = Lux.Chain(Lux.Dense(state, 30, tanh), Lux.Dense(30, state))
p_init, st = Lux.setup(rng, nn)

neuralode = NeuralODE(nn, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
prob_node = ODEProblem((u,p,t) -> nn(u,p,st)[1], [[u0[1];zeros(state-1)]], tspan, ComponentArray(p_init))

# Define parameters for Multiple Shooting
function loss_function(data, pred)
	return sum(abs2, data - pred)
end

function continuity_loss_function(u_end, u_0)
    return mean(abs2, u_end - u_0)
end

function loss_multiple_shooting(p)
    return multiple_shoot(p, Xₙ, tsteps, prob_node, loss_function, AutoTsit5(Rosenbrock23(autodiff=false)),
                          group_size; continuity_term)
end


function predict_final(θ)
    return Array(neuralode([u0[1]; zeros(state -1)], θ, st)[1])
end

function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = mean(abs2, Xₙ[1,:] .- X̂[1, :])
    prediction_error
end


function predict_single_shooting(p)
    return Array(neuralode([[u0[1];zeros(state-1)]], p, st)[1])
end


losses = Float32[]

callback = function (p, l, preds; doplot = false)
    push!(losses, final_loss(p))
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")

    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_init))
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters=5000; callback = callback)

function loss_single_shooting(p)
    pred = predict_single_shooting(p)
    l = loss_function(Xₙ, pred)
    return l, pred
end

full_traj = predict_final(res_ms.u)
full_traj_loss = final_loss(res_ms.u)
plot(full_traj[1, :])
scatter!(Xₙ[1, :])