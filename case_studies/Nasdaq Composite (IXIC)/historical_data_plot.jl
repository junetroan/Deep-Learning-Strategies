#Loading libraries
using CSV, DataFrames, Plots, Statistics, StatsBase
gr()

# Loading the data
data_path = "/Users/junetroan/Downloads/^IXIC (1).csv"
data = CSV.read(data_path, DataFrame)

# Extracting the relevant data
date = data[:, 1]
open_price = data[:, 2]

p = plot(date, open_price, title="Nasdaq Composite (IXIC) historical data", xlabel = "Date", ylabel = "Opening Price", color="#F21F98", label="Opening Price")
savefig("case_studies/Nasdaq Composite (IXIC)/Nasdaq Composite (IXIC) historical data.png")