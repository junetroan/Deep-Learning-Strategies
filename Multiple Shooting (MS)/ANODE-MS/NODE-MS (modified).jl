# MODIFIED NODE-MS MODEL

# Test againt Julias multiple shooting
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Distributions
using ComponentArrays, Lux, DiffEqFlux, Optimization, OptimizationPolyalgorithms, DifferentialEquations, Plots
using DiffEqFlux: group_ranges
using OptimizationOptimisers
using StableRNGs

gr()
rng = StableRNG(1111)
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

tspan = (0.0f0, 10.0f0)
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
println("solving ODE")
@time solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 1.0)
X = Array(solution)
x = X[1,:]

# SPECIFICATIONS
group_size = 5
state = 2
groupsize = 5
predsize = 5
tsteps = 1

U = Lux.Chain(Lux.Dense(state, 30, tanh),
              Lux.Dense(30, state))
p, st = Lux.setup(rng, U)

params = ComponentVector{Float32}(vector_field_model = p)
neuralode = NeuralODE(U, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
prob_node = ODEProblem((u,p,t) -> U(u, p, st)[1][1:end], u0, tspan, ComponentArray(p))


datasize = size(x, 1)
tspan = (0.0f0, 10.0f0) # Original (0.0f0, 10.0f0
tsteps = range(tspan[1], tspan[2]; length = datasize)
#solve(prob_node, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps)

