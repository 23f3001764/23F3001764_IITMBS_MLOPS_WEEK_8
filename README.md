# 🌸 Iris MLOps Pipeline

## 🎯 Objective  
This repository sets up an **MLOps pipeline** for the **Iris dataset** using GitHub with `main` and `dev` branches.  
It includes **pytest-based unit tests** for data validation and model evaluation, configured with **GitHub Actions for Continuous Integration (CI)**.  

The CI pipeline:  
- Fetches the **model** and **dataset** from **DVC** (stored in Google Cloud Storage).  
- Runs **tests** automatically on push or pull request.  
- Posts **sanity test results** as a PR/commit comment using **CML**.  
- Ensures only validated changes are merged from `dev` → `main`.

---

## 📁 Files and Utilities

### 🧪 `tests/test_validation_and_evaluation.py`
Pytest script for unit tests.  
Performs the following checks:
- Validates data file existence and structure.  
- Checks column integrity and absence of null values.  
- Verifies the presence of the model file.  
- Ensures prediction output shape is correct.  
- Confirms model accuracy ≥ **0.8** using metrics from `metrics.json`.  

✅ Ensures **data** and **model quality** before integration.

---

### ⚙️ `.github/workflows/ci.yml`
GitHub Actions workflow file for **Continuous Integration**.  

**Triggered on:**  
- Push to `main` or `dev`.  
- Pull request to `main`.  

**Performs the following steps:**
1. Authenticates to **Google Cloud**.  
2. Pulls data and model from **DVC remote**.  
3. Installs dependencies.  
4. Runs `pytest` tests.  
5. Saves results to `ci_results/result.txt`.  
6. Commits and pushes the results.  
7. Comments the test report on the commit or PR for visibility.  

---

### 📘 `MLOPS week 4.pdf`
Documentation outlining:
- Project goal and structure.  
- Step-by-step setup instructions (directory creation, file copying from GCS, Git initialization, secrets configuration, CI triggering, PR creation).  
- Explanation of the **CI YAML** and **test script**.  
- A detailed table of test checks.  

📖 Serves as a complete guide for **reproduction and understanding**.

---

### 🧾 `history.sh`
Bash script containing the **command history** of setup.  

Includes:
- Virtual environment activation.  
- DVC/GCS pulls.  
- Directory and file creation.  
- Git operations (`init`, `add`, `commit`, `push`, branch creation, PR, merge).  

🔍 Useful for **auditing** or **replicating** the exact setup process.

---

### 📄 `ci_results/result.txt`
Auto-generated output from the latest **pytest** run during CI.  

Contains:
- Test session details.  
- Passed/failed test summaries.  
- Warnings (e.g., sklearn feature name warnings).  

💡 Provides a **quick snapshot** of the CI test results without re-running the pipeline.

---

## 🚀 Summary
This repository demonstrates a **complete MLOps workflow** integrating:
- **Version control (Git + GitHub)**  
- **Data/model tracking (DVC)**  
- **Automated testing (Pytest)**  
- **Continuous Integration (GitHub Actions + CML)**  
- **Cloud storage (Google Cloud Storage)**  

It ensures reproducibility, maintainability, and quality control across the **machine learning lifecycle**.

---

### 🧠 Author
**Sahil Raj**  
📍 BS in Data Science, IIT Madras  
📅 October 2025
