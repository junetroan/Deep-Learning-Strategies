#ANODE-MS Model 2
using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
      OptimizationOptimisers, Random, Plots, StableRNGs , Plots
gr()

# Set a random seed for reproducible behaviour
rng = StableRNG(1111)

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Define the experimental parameter
tspan = (0.0f0, 10.0f0)
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
println("solving ODE")
@time solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)

# Add noise in terms of the mean
X = Array(solution)
t = solution.t
x̄ = mean(X, dims = 2)
noise_magnitude = 10f-3
noise_magnitude2 = 62.3f-2
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))
y = Xₙ[1,:]
state = 2
u0 = Float32[y[1];zeros(state-1)]

# Simple NN to predict lotka-volterra dynamics
U = Lux.Chain(Lux.Dense(state, 30, tanh), Lux.Dense(30, state))
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)
params = ComponentArray(p)
tsteps = range(tspan[1], tspan[2], length = length(y))
prob_neuralode = NeuralODE(U, tspan, Tsit5(); saveat = tsteps)

function predict(p)
    Array(prob_neuralode(u0, p, st)[1])
end

pr = predict(params)

function loss(p)
    pred = predict(p)
    loss = sum(abs2, y.- pred[1,:])
    loss
end

losses = Float32[]
callback = function (p, l)
    push!(losses, loss(p))
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(), callback=callback, maxiters = 5000)

y_pred = predict(res_ms.u)

plot(y_pred[1,:])
scatter!(y)

full_traj_loss = loss(res_ms.u)
println("Full Trajectory Loss: ", full_traj_loss)

optf_final = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
optprob_final = Optimization.OptimizationProblem(optf_final, res_ms.u)
res_final = Optimization.solve(optprob_final, BFGS(initial_stepnorm=0.01), maxiters = 1000, callback=callback, allow_f_increases = true)

print(losses[end])

full_traj2 = predict(res_final.u)
actual_loss = Xₙ[1,:] - full_traj2[1,:]
total_loss = abs(sum(actual_loss))

plot(full_traj2[1, :])
scatter!(y)
