import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Snack Portion Planner",
    layout="centered"
)

st.title("🍎 AI-Based Snack Portion Planner")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    xls = pd.ExcelFile("/content/Edited.xlsx")
    df_adults = pd.read_excel(xls, "Healthy_Adults_Meals")
    df_snacks = pd.read_excel(xls, "Snack_list_Cor")

    df_snacks = df_snacks.rename(columns={
        "Snack Type ": "Snack_Type",
        "Sub Type ": "Sub_Type",
        "Serving Size(g)/(ml) ": "Serving_Size",
        "g/ml": "Size_unit",
        "Car/g": "Carb_g",
        "Protein/g ": "Protein_g",
        "Fat/g": "Fat_g",
        "Energy/g (Kcal)": "Energy_kcal_g",
        "High energy/Medium enegy/Low enegy ": "Energy_Category"
    })

    return df_adults, df_snacks

df_adults, df_snacks = load_data()

# ======================
# TRAIN MODEL (ONCE)
# ======================
@st.cache_resource
def train_model(df):
    X = df[
        ['Gender', 'Age', 'Height (m)', 'Weight (kg)',
         'Daily_total_calory_intake_kcal', 'Income Level']
    ]

    y = df['Total_Calory_Requirement_kcal_(BMR*PhysicalActivity)']

    categorical = ['Gender', 'Income Level']
    numerical = ['Age', 'Height (m)', 'Weight (kg)',
                 'Daily_total_calory_intake_kcal']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(drop='first'), categorical)
    ])

    model = Pipeline([
        ('prep', preprocessor),
        ('ridge', Ridge(alpha=1.0))
    ])

    model.fit(X, y)
    return model

model = train_model(df_adults)

# ======================
# HELPER FUNCTIONS
# ======================
def food_row(food):
    return df_snacks[df_snacks['Name '].str.lower() == food.lower()]

def food_portion_atlas(food):
    row = food_row(food)
    if row.empty:
        return None

    s = row['Serving_Size'].values[0]
    e = row['Energy_kcal_g'].values[0]

    items = np.arange(1, 6)
    energy = items * s * e

    fig, ax = plt.subplots()
    ax.bar(items, energy)
    ax.set_xlabel("Number of items")
    ax.set_ylabel("Energy (kcal)")
    ax.set_title(f"Food Portion Atlas: {food}")
    return fig

def macro_chart(food):
    row = food_row(food)
    if row.empty:
        return None

    macros = [
        row['Carb_g'].values[0],
        row['Protein_g'].values[0],
        row['Fat_g'].values[0]
    ]

    fig, ax = plt.subplots()
    ax.pie(macros, labels=['Carb', 'Protein', 'Fat'],
           autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Macronutrient Distribution: {food}")
    return fig

# ======================
# USER INPUT
# ======================
st.header("👤 Personal Information")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age (years)", 18, 64, 25)
height = st.number_input("Height (m)", 1.3, 2.2, 1.65)
weight = st.number_input("Weight (kg)", 40.0, 120.0, 60.0)
daily_intake = st.number_input("Daily Calorie Intake (kcal)", 800, 4000, 1800)
income = st.selectbox("Income Level", [1, 2, 3])

bmi = weight / (height ** 2)

if bmi < 18.5:
    st.warning("Underweight – model intended for healthy adults.")
elif bmi > 22.9:
    st.warning("Overweight – model intended for healthy adults.")

# ======================
# PREDICTION
# ======================
if st.button("🔍 Calculate Requirement"):

    user = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height (m)': [height],
        'Weight (kg)': [weight],
        'Daily_total_calory_intake_kcal': [daily_intake],
        'Income Level': [income]
    })

    predicted = model.predict(user)[0]
    gap = predicted - daily_intake

    st.success(f"Predicted Requirement: {predicted:.1f} kcal")
    st.info(f"Calorie Gap: {gap:.1f} kcal")

    if gap <= 0:
        st.write("✅ No additional snack needed.")
        st.stop()

    # ======================
    # SNACK OPTIONS
    # ======================
    st.header("🍪 Snack Planning Options")

    option = st.selectbox(
        "Choose Option",
        [
            "1 – Portion size for a given snack",
            "2 – Suitability of given number of items",
            "3 – One fixed snack + one suggested snack",
            "4 – Multiple snacks until energy fulfilled",
            "5 – Multiple snacks with fixed quantities"
        ]
    )

    food_list = sorted(df_snacks['Name '].unique())

    food = st.selectbox("Select Snack", food_list)

    row = food_row(food)
    e = row['Energy_kcal_g'].values[0]
    s = row['Serving_Size'].values[0]
    unit = row['Size_unit'].values[0]

    st.pyplot(food_portion_atlas(food))
    st.pyplot(macro_chart(food))

    if option.startswith("1"):
        portion = gap / e
        items = portion / s
        st.write(f"Required portion: **{portion:.1f} {unit}**")
        st.write(f"Required items: **{items:.1f}**")

    elif option.startswith("2"):
        items = st.number_input("Items eaten", 0.0, 10.0, 1.0)
        energy = items * s * e
        diff = gap - energy
        st.write(f"Energy consumed: **{energy:.1f} kcal**")
        st.write(
            f"{'Remaining gap' if diff >= 0 else 'Excess intake'}: **{abs(diff):.1f} kcal**"
        )

    elif option.startswith("3"):
        items1 = st.number_input("Items of first snack", 0.0, 10.0, 1.0)
        used = items1 * s * e
        remaining = gap - used

        st.write(f"{food}: **{used:.1f} kcal**")

        if remaining > 0:
            food2 = st.selectbox("Second Snack", food_list)
            row2 = food_row(food2)
            e2 = row2['Energy_kcal_g'].values[0]
            s2 = row2['Serving_Size'].values[0]
            unit2 = row2['Size_unit'].values[0]

            portion2 = remaining / e2
            items2 = portion2 / s2

            st.write(
                f"{food2} portion: **{portion2:.1f} {unit2} ({items2:.1f} items)**"
            )

    else:
        st.info("Options 4 & 5 are iterative by design — recommend converting to repeated button clicks.")
