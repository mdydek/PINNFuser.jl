---

# Project Name
A brief description of the project goes here.


## Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/mdydek/inzynierka.git
```
Navigate to this directory:
```bash
cd inzynierka
```

Start Julia and enter the package manager by typing `]`:
```julia
activate .
```
This will activate the project's environment. 

Install all dependencies using:
```julia
instantiate
```

Now the project is ready to be run or developed further:

## Usage
Leave the package manager by pressing BACKSPACE or Ctrl + C. This will take you back to the Julia REPL.
```julia
include("src/1.jl")
```

## Adding Packages
To add other packages to this project, you can use:
```julia
add PackageName
```
Replace `PackageName` with the packages you wish to add.

---
