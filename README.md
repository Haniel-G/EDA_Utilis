---
<h3 align="center">
  <em><strong>Code being reworked ⚠️</strong></em>
</h3>

---

# Exploratory Data Analysis (EDA) - Utilis

This project is a utility of functions that assist in database EDA, and is constantly being improved.

## 1. Description  
**EDA Utilities** is a reusable and scalable Python module designed to streamline **Exploratory Data Analysis (EDA)** across various domains. This toolkit simplifies **data loading, cleaning, visualization, and statistical analysis**, making the initial phases of any data science project more efficient and structured.  

With built-in **logging** and **custom exception handling**, the module ensures **reliable debugging and error tracking**, enabling seamless data exploration for datasets in **finance, healthcare, marketing, and more**.  

## 2. Technologies and Tools  
This project leverages **Python** and key **data science libraries**. The technologies and tools used were Python (Pandas, Numpy, Matplotlib, Seaborn, logging and os), Jupyter Notebook, Git and Github (version control), statistics, code runner (terminal), and Visual Studio Code (project development environment).

## 3. Project Objective  

This EDA framework was built to **accelerate** the data exploration phase while maintaining **data integrity and insights generation**.  

- **Automated Data Loading**: Easily handle .csv and .xlsx files  
- **Data Cleaning & Transformation**: Handle missing values, convert dates, and format numerical values  
- **Statistical Insights**: Generate descriptive statistics, detect outliers, and compute variability metrics  
- **Powerful Visualizations**: Create histograms, boxplots, and bar charts with minimal effort  
- **Error Handling & Logging**: Ensure smooth debugging and structured exception reporting  
With this package, **data scientists can focus on insights rather than preprocessing**.

## 4. Next Steps  
This project is continuously evolving. The following features are planned for future versions:

- **Data Type Detection & Auto-Cleaning** - Automate the identification and handling of categorical, numerical, and datetime features.
- **Outlier Handling Options** - Expand strategies beyond IQR, such as Z-score and Isolation Forests.
- **Expand the project to be a general data science tool** - I also intend to develop utilities for machine learning modeling.

## 5. Installation 📾  
### Requirements:  
- Python (3.13.0)  
- pip (25.0.1)  
- Git (version control tool)  

Once you have these installed, open a terminal on your local machine and run the following commands:

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/Haniel-G/EDA_Utilis.git
   ```

2. **Navigate to the cloned repository directory:**  
   ```bash
   cd EDA_Utilis
   ```

3. **Create a virtual environment:**  
   ```bash
   python -m venv nome_da_venv
   ```

4. **Activate the virtual environment:**  
   ```bash
   source .venv/Scripts/activate  # On Linux, use 'venv/bin/activate'
   ```

5. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

### 5.1 Run with application  
To integrate **EDA Utilis** into your existing project and ensure all dependencies are installed correctly, follow these steps:

1. **Clone the repository (if not already cloned):**  
   ```bash
   git clone https://github.com/Haniel-G/EDA_Utilis.git
   ```

2. **Navigate to the cloned repository directory:**  
   ```bash
   cd EDA_Utilis
   ```

3. **Merge dependencies with your existing project:**  
   If you already have a `requirements.txt` file in your project, append the dependencies from **EDA Utilis** without removing your existing ones:  
   ```bash
   cat requirements.txt >> ../requirements.txt
   ```
   Then, remove duplicate entries to avoid conflicts:  
   ```bash
   sort -u ../requirements.txt -o ../requirements.txt
   ```

4. **Install the updated dependencies:**  
   ```bash
   pip install -r ../requirements.txt
   ```

5. **Import and use EDA Utilis in your project:**  
   After installation, you can import and utilize its functions in your Python scripts or Jupyter notebooks:  
   ```python
   from src.eda_utilis import *
   ```


