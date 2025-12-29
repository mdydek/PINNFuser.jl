using Pkg
Pkg.instantiate()

using JuliaFormatter

is_formatted = format(".", verbose = false, overwrite = false);
if !is_formatted
    println("❌ Some files are not formatted. Run JuliaFormatter and commit changes.")
    exit(1)
end
