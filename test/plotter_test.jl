using Test
using Plots
using DelimitedFiles

include("../lib/lib.jl")
using .LibInfuser


@testset "PINNPlotter Test Suite" begin

    @testset "plot_PINN_results Functionality" begin
        samples = 20
        equations = 2

        infused = rand(samples, equations)
        data = rand(samples, equations)
        ode_sol = rand(samples, equations)

        data_steps = collect(range(0, 10, length = samples))
        extrap_steps = collect(range(0, 10, length = samples))

        labels_correct = ["Data 1", "PINN 1", "ODE 1", "Data 2", "PINN 2", "ODE 2"]

        labels_incorrect = ["Data 1", "PINN 1"]

        filename = "test_prediction_plot.png"

        if isfile(filename)
            rm(filename)
        end

        LibInfuser.PINNPlotter.plot_PINN_results(
            infused,
            data,
            ode_sol,
            labels_correct,
            data_steps,
            extrap_steps,
            "Time",
            "Value",
            "Test Title",
            filename,
        )

        @test isfile(filename)
        rm(filename)

        @test_throws BoundsError LibInfuser.PINNPlotter.plot_PINN_results(
            infused,
            data,
            ode_sol,
            labels_incorrect,
            data_steps,
            extrap_steps,
            "Time",
            "Value",
            "Fail Title",
            "fail.png",
        )
    end

    @testset "plot_loss Functionality" begin
        loss_filename = "test_loss_history.txt"
        plot_filename = "test_loss_plot.png"

        open(loss_filename, "w") do io
            println(io, "# Iteration Loss")
            println(io, "1 0.99")
            println(io, "2 0.50")
            println(io, "3 0.10")
            println(io, "")
            println(io, "4 0.01")
        end

        if isfile(plot_filename)
            rm(plot_filename)
        end

        LibInfuser.PINNPlotter.plot_loss(
            loss_filename;
            xlabel = "Iter",
            ylabel = "Err",
            title = "Test Loss",
            plotfile = plot_filename,
            logscale = false,
        )

        @test isfile(plot_filename)
        rm(plot_filename)

        LibInfuser.PINNPlotter.plot_loss(
            loss_filename;
            plotfile = plot_filename,
            logscale = true,
        )

        @test isfile(plot_filename)
        rm(plot_filename)
        rm(loss_filename)
    end

    @testset "Directory Creation" begin
        loss_filename = "temp_loss.txt"
        open(loss_filename, "w") do io
            println(io, "1 1.0")
        end

        subdir = "test_plots_subdir"
        plot_path = joinpath(subdir, "loss.png")

        if isdir(subdir)
            rm(subdir, recursive = true)
        end

        LibInfuser.PINNPlotter.plot_loss(loss_filename, plotfile = plot_path)

        @test isdir(subdir)
        @test isfile(plot_path)

        rm(subdir, recursive = true)
        rm(loss_filename)
    end

end
