# **ChronoSwarm**

This project provides a scheduling optimization algorithm based on MUulti-Swarm Particle Swarm Optimization (MSPSO)

---


## **Requirements**
- Python 3.8 or later
- `ctt-validator.exe` for output validation // already in 'mnt/data'

---

## **Installation Steps**
1. **Install Python**: Ensure Python is installed on your system. If not, download and install it from [python.org](https://www.python.org/). Make sure to check the "Add Python to PATH" option during installation.
2. **Navigate to Project Directory**: Use the terminal or Command Prompt to move to the folder containing the `requirements.txt` file.  
   ```bash
   cd path/to/project
   ```
3. **Install Dependencies**: Run the following command to install all required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## **How It Works**

### **Running the Code**
1. **Execute the Solver**:
   Run the `solver.py` file to start the optimization process:
   ```bash
   python solver.py
   ```

2. **Change Instances**:
   To solve a different instance, modify the `comp` variable in the `config.py` file. Save your changes and re-run the solver.

3. **Validate Outputs**:
   To validate the generated schedule:
   - Open the terminal and navigate to the `mnt\data` directory.
   - Run the following command:
     ```bash
     ctt-validator.exe 'name_of_comp'.ctt 'name_of_comp'.out
     ```
   Replace `'name_of_comp'` with the actual name of the competition or instance file.

---

## **Configuration**
- Adjust which instance to solve in `config.py`.

---

## **Folder Structure**
```plaintext
.
├── config.py                  # Configuration file for PSO parameters and instances
├── ctt_parser.py              # Parses CTT files for problem input
├── ctt.pdf                    # Documentation on the CTT problem format
├── initialize_population.py   # Initializes the particle (Graph Based Heuristic)
├── multiswarm.py              # Multi-swarm PSO implementation
├── readme.md                  # Documentation
├── requirements.txt           # Python dependencies
├── solver.py                  # Main script to run the optimization algorithm
└── mnt/data                   # Folder for Input and Output files
    	└── ctt-validator.exe      # Validation tool for the output files
```

## **Support**
For assistance, contact gdrenomeron@up.edu.ph or renomerong4@gmail.com

