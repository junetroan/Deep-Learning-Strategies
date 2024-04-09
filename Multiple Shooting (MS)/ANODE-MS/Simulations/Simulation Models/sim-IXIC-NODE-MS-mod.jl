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
using CSV, Tables, DataFrames
using StatsBase
gr()

data_path = "Multiple Shooting (MS)/ANODE-MS/Data/^IXIC.csv"
data = CSV.read(data_path, DataFrame)
datasize = length(data[:, 1])

split_ratio = 0.3
train = data[1:Int(round(split_ratio * datasize)), 2]
test = data[Int(round(split_ratio * datasize)):end, 2]

date = data[:, 1]
open_price = data[:, 2]
high_price = data[:, 3]
low_price = data[:, 4]
close_price = data[:, 5]
adj_close = data[:, 6]
volume = data[:, 7]

t = collect(1:datasize)
t_train = collect(1:Int(round(split_ratio * datasize)))
t_test = collect(Int(round(split_ratio * datasize)):datasize)

transformer = fit(ZScoreTransform, open_price)
X_train = Float32.(StatsBase.transform(transformer, train))
X_test = Float32.(StatsBase.transform(transformer, test))

x = X_train

# SPECIFICATIONS
group_size = 5
state = 2
groupsize = 5
predsize = 5
tsteps = 1
rng = StableRNG(1111)

tspan = (t_train[1], t_train[end])
datasize = size(x, 1)
tsteps = range(tspan[1], tspan[2]; length = datasize)
u0 = Float32[X_train[1], mean(X_train[1:groupsize])]

U = Lux.Chain(Lux.Dense(state, 30, tanh),
              Lux.Dense(30, state))
p, st = Lux.setup(rng, U)

params = ComponentVector{Float32}(vector_field_model = p)
neuralode = NeuralODE(U, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
prob_node = ODEProblem((u,p,t) -> U(u, p, st)[1][1:end], u0, tspan, ComponentArray(p))

ranges = group_ranges(datasize, group_size)
u0 = Float32(x[first(1:5)])
u0_init = [[x[first(rg)]; fill(mean(x[rg]), state - 1)] for rg in ranges] 
u0_init = mapreduce(permutedims, vcat, u0_init)

function multiple_shoot_mod(p, ode_data, tsteps, prob::ODEProblem, loss_function::F,
    continuity_loss::C, solver::SciMLBase.AbstractODEAlgorithm, group_size::Integer;
    continuity_term::Real = 100, kwargs...) where {F, C}

    datasize = size(ode_data, 1)

    if group_size < 2 || group_size > datasize
        throw(DomainError(group_size, "group_size can't be < 2 or > number of data points"))
    end

    ranges = group_ranges(datasize, group_size)

    sols = [solve(remake(prob_node; p = p.θ, tspan = (tsteps[first(rg)], tsteps[last(rg)]),
        u0 = p.u0_init[index, :]),
       solver; saveat = tsteps[rg]) 
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
continuity_term = 10.0
params = ComponentVector{Float32}(θ = p, u0_init = u0_init)

# Modified multiple_shoot method 
multiple_shoot_mod(params, x, tsteps, prob_node, loss_function,
    continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size;
    continuity_term)

function loss_multiple_shooting(p)
    return multiple_shoot_mod(p, x, tsteps, prob_node, loss_function, continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size; continuity_term)
end


loss_multiple_shooting(params) 

losses = Float32[]

callback = function (p, l, preds; doplot = false)
    push!(losses, loss_multiple_shooting(p)[1])
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")

    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf,params)
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters=5000; callback = callback)

losses[end]