import subprocess

# List all your scripts in the order you want to run them
scripts = [
    "gold.py",
    "check missing value.py",
    "Basic Stat Check.py",
    "RF Model1.py",
    "rf model2.py"
]

with open("all_results.txt", "w") as outfile:
    for script in scripts:
        # Run each script and append the output to all_results.txt
        # The 'text=True' ensures the output is treated as a string.
        subprocess.run(["python", script], stdout=outfile, stderr=outfile, text=True)

print("All results have been written to all_results.txt")