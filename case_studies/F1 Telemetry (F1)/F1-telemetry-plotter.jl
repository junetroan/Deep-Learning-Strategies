using CSV, DataFrames, Plots, Statistics, StatsBase
using GR
gr(fontfamily = "Computer Modern", legendfontsize = 10, guidefontsize = 10, tickfontsize = 10)

# Loading the Data
path = "case_studies/F1 Telemetry (F1)/data/telemetry_data_LEC_spain_2023_qualifying.csv"
data = CSV.read(path, DataFrame)
distance = data.Distance
speed = data.Speed
throttle = data.Throttle
brake = data.Brake

# Create a figure with three subplots
Plots.plot(layout = (3, 1), size=(800, 600))

# Plot Speed
Plots.plot!(distance, speed, label="Speed", color="#F21F98", xlabel="Distance", ylabel="Speed", subplot=1)

# Plot Throttle
Plots.plot!(distance, throttle, label="Throttle", color="#F5A623", xlabel="Distance", ylabel="Throttle", subplot=2)

# Plot Brake (assuming brake is boolean, plot it as a binary value)
Plots.plot!(distance, brake, label="Brake", color="#A571DC", xlabel="Distance", ylabel="Brake", subplot=3, seriestype=:step)

# Save the plot to a file
Plots.savefig("case_studies/F1 Telemetry (F1)/F1 Telemetry.png")