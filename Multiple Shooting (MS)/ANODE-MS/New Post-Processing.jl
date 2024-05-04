using CSV
using DataFrames
using Plots
using Statistics
using

# Set the backend for Plots to Plotly
plotly()

# Function to read CSV files into a DataFrame
function read_data(folder_path)
    df = DataFrame()
    file_names = readdir(folder_path)[1:400]  # Adjust the range as necessary
    for file_name in file_names
        full_path = joinpath(folder_path, file_name)
        col_data = CSV.read(full_path, DataFrame)
        col_name = splitext(basename(file_name))[1]
        df[!, Symbol(col_name)] = col_data[!, 1]  # Assuming data in the first column
    end
    return df
end

# Plotting function for loss evolution
function plot_loss(df, title)
    p = plot(title=title, xlabel="Iterations", ylabel="Loss")
    for col_name in names(df)
        plot!(p, df[!, col_name], label=String(col_name))
    end
    #display(p)
    Plots.savefig(p, title * ".png")
end

# Averaging and plotting function with standard deviation
function average_loss_plot(df, title)
    row_means = mean.(eachrow(df))
    row_stds = std.(eachrow(df))
    p = plot(1:nrow(df), row_means, ribbon=row_stds, label="Average Loss with Std Dev", title=title, xlabel="Iteration", ylabel="Average Loss", legend=:bottomright, yscale=:log10)
    #display(p)
    Plots.savefig(p, title * ".png")
end

# Directory paths for data
folder_path_ANODE = "/Users/junetroan/Desktop/data-ANODEMS-LV-correct/Loss Data"
folder_path_NODE = "/Users/junetroan/Desktop/ModifiedMS_LV/Loss Data"
folder_path_PEM = "/Users/junetroan/Desktop/Results/files Ks/Loss Data Ks"

# Data reading
df_ANODE = read_data(folder_path_ANODE)
df_NODE = read_data(folder_path_NODE)
df_PEM = read_data(folder_path_PEM)  # Assuming you want a specific range here

# Plotting loss evolution
plot_loss(df_ANODE, "Loss Evolution of ANODE-MS II Model")
plot_loss(df_NODE, "Loss Evolution of MNODE-MS Model")
plot_loss(df_PEM, "Loss Evolution of NPEM Model")

# Average loss plots
average_loss_plot(df_ANODE, "Average Loss Evolution of ANODE-MS II Model (logscale)")
average_loss_plot(df_NODE, "Average Loss Evolution of MNODE-MS Model (logscale)")
average_loss_plot(df_PEM, "Average Loss Evolution of NPEM Model (logscale)")

function compare(df1, df2)

    row_means_1 = mean.(eachrow(df1))
    row_means_2 = mean.(eachrow(df2))

    row_stds_1 = std.(eachrow(df1))    
    row_stds_2 = std.(eachrow(df2))

    log_means_1 = log.(row_means_1)
    log_means_2 = log.(row_means_2)

    log_stds_1 = log.(row_stds_1)
    log_stds_2 = log.(row_stds_2)

    p = plot(1:nrow(df1), row_means_1, ribbon=row_stds_1, label="MNODE-MS", title="Comparison of Loss Evolution", xlabel="Iteration", ylabel="Average Loss")
        plot!(p, 1:nrow(df2), row_means_2, ribbon=row_stds_2, label="NPEM")
    
    #p2 = plot(1:nrow(df1), log_means_1, ribbon=log_stds_1, label="ANODE-MS", title="Comparison of Loss Evolution", xlabel="Iteration", ylabel="Average Loss")
    p3 = plot(1:nrow(df1), row_means_1, ribbon = log_stds_1, label="MNODE-MS", title="Comparison of Loss Evolution", xlabel="Iteration", ylabel="Average Loss", yscale=:log10)
    plot!(p3, 1:nrow(df2), row_means_2, ribbon = log_stds_2, label="NPEM")
    #plot!(p2, 1:nrow(df2), log_means_2, ribbon=log_stds_2, label="PEM")
    
    #display(p)
    #display(p2)
    #display(p3)
    
    #savefig(p, "Multiple Shooting (MS)/ANODE-MS/Simulations/Comparisons/LV-MS/Comparison ANODE-MS vs. PEM.png")
    #p3 = plot(1:nrow(df1), row_means_1, ribbon = log_stds_1, label="ANODE-MS", title="Comparison of Loss Evolution", xlabel="Iteration", ylabel="Average Loss", yscale=:log10, size=(800,600))
    #plot!(p3, 1:nrow(df2), row_means_2, ribbon = log_stds_2, label="PEM")
    
    #Plots.savefig(p, "Results/LV/Comparison ANODE-MS vs. NPEM.png")
    #Plots.savefig(p3, "Results/LV/Comparison ANODE-MS vs. NPEM (Log Scale).png")
    
    #Plots.savefig(p, "Results/LV/Comparison ANODE-MS vs. MNODE-MS.png")
    #Plots.savefig(p3, "Results/LV/Comparison ANODE-MS vs. MNODE-MS (Log Scale).png")

    #Plots.savefig(p, "Results/LV/Comparison MNODE-MS vs. NPEM.png")
    Plots.savefig(p3, "Results/LV/Comparison MNODE-MS vs. NPEM (Log Scale).png")
end

compare(df_ANODE, df_PEM)
compare(df_ANODE, df_NODE)
compare(df_NODE, df_PEM)