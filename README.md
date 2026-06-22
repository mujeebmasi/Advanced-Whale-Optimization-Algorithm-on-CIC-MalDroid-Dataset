# 🐋 Enhanced Whale Optimization Algorithm (EWOA)

An enhanced implementation of the **Whale Optimization Algorithm (WOA)**, a nature-inspired metaheuristic optimization technique based on the bubble-net hunting strategy of humpback whales. This project introduces several improvements to the standard WOA to achieve faster convergence, better exploration, and improved solution quality on optimization problems.

---

## 📌 Overview

The Whale Optimization Algorithm (WOA) is a population-based optimization algorithm proposed by Mirjalili and Lewis (2016). While the original WOA performs well on many optimization tasks, it may suffer from:

* Premature convergence
* Local optima stagnation
* Slow convergence on complex search spaces

This project implements an **Enhanced Whale Optimization Algorithm (EWOA)** with modifications designed to improve exploration and exploitation balance.

---

## ✨ Features

* ✅ Enhanced exploration strategy
* ✅ Improved exploitation mechanism
* ✅ Faster convergence speed
* ✅ Reduced risk of local optima trapping
* ✅ Benchmark function evaluation
* ✅ Visualization of convergence curves
* ✅ Modular and extensible code structure

---

## 🧠 Algorithm Workflow

1. Initialize whale population randomly.
2. Evaluate fitness of each whale.
3. Identify the current best solution.
4. Update whale positions using:

   * Encircling prey mechanism
   * Bubble-net attacking strategy
   * Random search exploration
5. Apply enhancement strategies.
6. Recalculate fitness values.
7. Update global best solution.
8. Repeat until termination criteria are met.



---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Enhanced-Whale-Optimization-Algorithm.git

cd Enhanced-Whale-Optimization-Algorithm
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the optimization algorithm:

```bash
python main.py
```

Example:

```python
from src.ewoa import EnhancedWOA

optimizer = EnhancedWOA(
    population_size=30,
    max_iterations=500,
    dimensions=30
)

best_solution, best_fitness = optimizer.optimize()

print(best_solution)
print(best_fitness)
```

---

## 📊 Experimental Results

The Enhanced WOA was evaluated on standard benchmark optimization functions.

| Function   | WOA    | Enhanced WOA |
| ---------- | ------ | ------------ |
| Sphere     | Better | Best         |
| Rastrigin  | Better | Best         |
| Rosenbrock | Better | Best         |
| Ackley     | Better | Best         |

### Convergence Curve

Add your convergence plot here:

```markdown
![Convergence Curve](plots/convergence.png)
```

---

## 🔬 Enhancement Techniques

This implementation introduces:

* Adaptive parameter control
* Dynamic exploration-exploitation balancing
* Diversity preservation mechanism
* Improved position update strategy
* Random perturbation for escaping local optima

---

## 📈 Applications

EWOA can be applied to:

* Machine Learning Hyperparameter Optimization
* Feature Selection
* Engineering Design Optimization
* Resource Allocation
* Scheduling Problems
* Energy Management Systems
* Image Processing Tasks

---

## 🛠 Technologies Used

* Python
* NumPy
* Matplotlib
* Pandas
* Scikit-Learn

---

## 📚 References

1. Mirjalili, S., & Lewis, A. (2016).
   *The Whale Optimization Algorithm.*
   Advances in Engineering Software, 95, 51–67.

2. Various recent enhancements and hybrid metaheuristic optimization studies.

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Abdul Mujeeb**

B.Tech Student | AI & Machine Learning Enthusiast

GitHub: https://github.com/yourusername
