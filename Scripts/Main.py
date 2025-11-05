import subprocess

print("Running Rosetta Analysis script...")
subprocess.run(["python", r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Scripts\Subscripts\Rosetta_Interface_Analysis.py"], check=True)

print("Normalising and scoring Rosetta and PISA results...")
subprocess.run(["python", r"C:\Users\aszyk\PycharmProjects\Miniproject 2 (CDR hallucination)\Scripts\Subscripts\Normalise and Score Rosetta and PISA.py"], check=True)

print("All scripts finished successfully!")
