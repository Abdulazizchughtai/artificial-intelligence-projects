# Group Names
**Abdulaziz Chughtai**, **Summaiya Sabir**, **Zain Ammad Khan**

# SAP IDs
**70137089**, **70137342**, **70138187**

---

# Student Performance Prediction System

This project is a Next.js web application developed to predict student academic performance and provide insights to help improve educational outcomes. In today's competitive academic environment, early identification of at-risk students is crucial. This system focuses on analyzing student behavioral and socio-economic data to present predictions and actionable recommendations through a modern, responsive web interface.

The application is built using **Next.js**, **Python (FastAPI)**, and **Tailwind CSS**, ensuring high performance, scalability, and an intuitive user experience.

## Student Name(s)
- **Abdulaziz Chughtai** (70137089)
- **Summaiya Sabir** (70137342)
- **Zain Ammad Khan** (70138187)

## Introduction

Academic failure often comes as a surprise to educators and parents, not because signs were missing, but because they were scattered across different aspects of a student's life. Predicting academic performance involves more than just looking at past grades; it requires a holistic view of study habits, attendance, and support systems.

This project provides a web-based solution that allows educators to input student profiles, estimate future grades, and receive personalized recommendations for improvement. By leveraging machine learning models trained on synthetic but realistic educational data, the system helps stakeholders make informed decisions to foster student success.

## Problem Statement

The key objectives of this project are:

1.  **Academic Performance Prediction**
    Analyze student data (study hours, attendance, etc.) to estimate future grades and pass/fail outcomes.

2.  **Risk Assessment**
    Identify students who are at "High", "Medium", or "Low" risk of faltering, allowing for prioritized intervention.

3.  **Actionable Recommendations**
    Provide specific, data-driven advice (e.g., "Increase study time by 5 hours", "Seek family support") to help improve the student's trajectory.

## Existing Solutions

Traditional academic monitoring typically relies on:
-   Mid-term exam results (often too late for major changes)
-   Manual observation by teachers (subjective and inconsistent)
-   Generic advice ("Study harder") without specific targets

These methods lack real-time predictive capabilities and personalization. While some Leaning Management Systems (LMS) offer analytics, they are often complex and not accessible to all educators or parents.

## Application Architecture & Methodology

This project follows a structured full-stack development approach, combining a powerful ML backend with a modern frontend.

### Frontend Framework
The application is built using **Next.js**, utilizing:
-   **Next.js App Router** for modern routing
-   **TypeScript** for type safety
-   **Tailwind CSS** for responsive, glass-morphism UI design
-   **Chart.js** for dynamic data visualization

### User Interface Features
The web interface allows users to:
-   Input student parameters (Hours studied, Attendance, Parent Education, etc.)
-   View predicted Percentage, Grade (A-F), and Risk Level
-   Analyze feature contributions via interactive charts
-   Receive tailored text-based recommendations

### Data Handling & Analysis
The system is designed around key educational indicators.
-   **Dataset Source**: Synthetically generated module (`backend/predictor.py`) simulating realistic scenarios.
-   **Key Features**: Study Hours, Previous Grade, Attendance, Assignments Completed, Extracurricular Activities, Internet Access, etc.

### Prediction & Insights
Based on the input data:
1.  **Estimated Grade** is calculated using a weighted scoring algorithm.
2.  **Pass/Fail Probability** is determined by the specific ML model.
3.  **Recommendations** are logic-based, deriving from specific deficits in the input profile.

## Model / Technique Used

We utilized a specific set of Machine Learning algorithms to ensure robust prediction capabilities. The system trains and evaluates:
-   **Logistic Regression**
-   **K-Nearest Neighbors (KNN)**
-   **Decision Tree Classifier**
-   **Random Forest Classifier** (Best Performing Model)

**Metrics**: Accuracy (>85%), AUC-ROC analysis for model comparison.

## Getting Started

### Prerequisites
-   Python 3.8+
-   Node.js 18+

### Installation & Run

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd section-e-projects/student-performance-prediction
    ```

2.  **Start the Backend**:
    ```bash
    cd backend
    pip install -r requirements.txt
    python main.py
    ```
    *Server runs at `http://localhost:8000`*

3.  **Start the Frontend**:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```
    *App runs at `http://localhost:3000`*

## Results
The model successfully differentiates between high-performing and at-risk students. The dashboard visualizes these results clearly.

**Screenshots**:
![UI Preview](ui-preview.png)

## References
-   [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
-   [Next.js Documentation](https://nextjs.org/docs)
-   [FastAPI Documentation](https://fastapi.tiangolo.com/)
