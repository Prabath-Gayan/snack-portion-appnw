import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

st.set_page_config(page_title="Snack Portion Planner", layout="centered")

# ======================
# SESSION STATE
# ======================
if "predicted" not in st.session_state:
    st.session_state.predicted = None
if "gap" not in st.session_state:
    st.session_state.gap = None

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    xls = pd.ExcelFile("Edited.xlsx")
    df_adults = pd.read_excel(xls, "Healthy_Adults_Meals")
    df_snacks = pd.read_excel(xls, "Snack_list_Cor")
    return df_adults, df_snacks

df_adults, df_snacks = load_data()

# Rename snack columns
df_snacks = df_snacks.rename(columns={
    "Snack Type ": "Snack_Type",
    "Sub Type ": "Sub_Type",
    "Serving Size(g)/(ml) ": "Serving_Size",
    "g/ml": "Size_unit",
    "Car/g": "Carb_g",
    "Protein/g ": "Protein_g",
    "Fat/g": "Fat_g",
    "Energy/g (Kcal)": "Energy_kcal_g",
    "High energy/Medium enegy/Low enegy ": "Energy_Category",
    "Name ": "Name"
})

# ======================
# TRAIN MODEL
# ======================
@st.cache_resource
def train_model():
    X = df_adults[
        ['Gender', 'Age', 'Height (m)', 'Weight (kg)',
         'Daily_total_calory_intake_kcal', 'Income Level']
    ]
    y = df_adults['Total_Calory_Requirement_kcal_(BMR*PhysicalActivity)']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(),
             ['Age', 'Height (m)', 'Weight (kg)', 'Daily_total_calory_intake_kcal']),
            ('cat', OneHotEncoder(drop='first'),
             ['Gender', 'Income Level'])
        ]
    )

    model = Pipeline([
        ('prep', preprocessor),
        ('ridge', Ridge(alpha=1.0))
    ])

    model.fit(X, y)
    return model

model = train_model()

# ======================
# UI
# ======================
st.title("🍎 Personalized Snack Portion Planner")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 64, 25)
height = st.number_input("Height (m)", 1.0, 2.5, 1.65)
weight = st.number_input("Weight (kg)", 30.0, 150.0, 60.0)
daily_intake = st.number_input("Daily Calorie Intake (kcal)", 1000, 4000, 1800)
income = st.selectbox("Income Level", [1, 2, 3])

# ======================
# STEP 1: PREDICT
# ======================
if st.button("Predict Requirement"):
    bmi = weight / (height ** 2)

    if bmi < 18.5 or bmi > 22.9:
        st.error("This model is intended for healthy adults (BMI 18.5–22.9).")
        st.session_state.predicted = None
        st.session_state.gap = None
    else:
        user = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Height (m)': [height],
            'Weight (kg)': [weight],
            'Daily_total_calory_intake_kcal': [daily_intake],
            'Income Level': [income]
        })

        pred = model.predict(user)[0]
        gap = pred - daily_intake

        st.session_state.predicted = pred
        st.session_state.gap = gap

# ======================
# DISPLAY PREDICTION
# ======================
if st.session_state.predicted is not None:
    st.success(f"Predicted Requirement: {st.session_state.predicted:.1f} kcal/day")
    st.info(f"Calorie Gap: {st.session_state.gap:.1f} kcal")

    if st.session_state.gap <= 0:
        st.info("No additional snack needed.")

# ======================
# STEP 2 & 3: SNACK + PORTION
# ======================
if st.session_state.gap is not None and st.session_state.gap > 0:

    snack = st.selectbox("Choose a snack", df_snacks["Name"].unique())

    if st.button("Calculate Portion"):
        row = df_snacks[df_snacks["Name"] == snack].iloc[0]

        portion = st.session_state.gap / row["Energy_kcal_g"]
        items = portion / row["Serving_Size"]

        st.write(f"### Suggested Portion for {snack}")
        st.write(f"• {portion:.1f} {row['Size_unit']}")
        st.write(f"• {items:.1f} items")

        # Energy bar graph
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3])
        y = x * row["Serving_Size"] * row["Energy_kcal_g"]

        ax.bar(x, y)
        ax.set_xlabel("Number of Items")
        ax.set_ylabel("Energy (kcal)")
        ax.set_title("Energy vs Portion Size")

        st.pyplot(fig)

        # ======================
        # MACRONUTRIENT PIE CHART
        # ======================
        macros = {
            "Carbohydrates (g)": row["Carb_g"],
            "Protein (g)": row["Protein_g"],
            "Fat (g)": row["Fat_g"]
        }

        fig2, ax2 = plt.subplots()
        ax2.pie(macros.values(), labels=macros.keys(), autopct="%1.1f%%", startangle=90,
                colors=["#FFD700", "#87CEFA", "#FF6347"])
        ax2.set_title(f"Macronutrient Distribution of {snack}")
        ax2.axis("equal")  # Equal aspect ratio ensures pie is circular

        st.pyplot(fig2)
