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
defaults = {
    "predicted": None,
    "gap": None,
    "total_consumed": 0.0,
    "option4_list": [],
    "option5_list": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
# MODEL
# ======================
@st.cache_resource
def train_model():
    X = df_adults[
        ['Gender', 'Age', 'Height (m)', 'Weight (kg)',
         'Daily_total_calory_intake_kcal', 'Income Level']
    ]
    y = df_adults['Total_Calory_Requirement_kcal_(BMR*PhysicalActivity)']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(),
         ['Age', 'Height (m)', 'Weight (kg)', 'Daily_total_calory_intake_kcal']),
        ('cat', OneHotEncoder(drop='first'),
         ['Gender', 'Income Level'])
    ])

    model = Pipeline([
        ('prep', preprocessor),
        ('ridge', Ridge(alpha=1.0))
    ])
    model.fit(X, y)
    return model

model = train_model()

# ======================
# VISUALS
# ======================
def food_portion_atlas(row, max_items=5):
    items = np.arange(1, max_items + 1)
    energy = items * row["Serving_Size"] * row["Energy_kcal_g"]
    fig, ax = plt.subplots()
    ax.bar(items, energy)
    ax.set_xlabel("Number of Items")
    ax.set_ylabel("Energy (kcal)")
    ax.set_title(row["Name"])
    st.pyplot(fig)

def macro_pie_chart(row):
    macros = {
        "Carbohydrate": row["Carb_g"],
        "Protein": row["Protein_g"],
        "Fat": row["Fat_g"]
    }
    fig, ax = plt.subplots()
    ax.pie(macros.values(), labels=macros.keys(), autopct="%1.1f%%")
    ax.set_title("Macronutrient Distribution")
    st.pyplot(fig)

# ======================
# UI
# ======================
st.title("🍎 Personalized Snack Portion Planner")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 64, 25)
height = st.number_input("Height (m)", 1.0, 2.5, 1.65)
weight = st.number_input("Weight (kg)", 30.0, 150.0, 60.0)
daily_intake = st.number_input("Daily Calorie Intake (kcal)", 1000, 4000, 1800)
income = st.selectbox("Income Level", [1, 2, 3])

# ======================
# PREDICT
# ======================
if st.button("Predict Requirement"):
    bmi = weight / height**2
    if not (18.5 <= bmi <= 22.9):
        st.error("BMI must be between 18.5 and 22.9")
        st.session_state.predicted = None
    else:
        user = pd.DataFrame({
            "Gender": [gender],
            "Age": [age],
            "Height (m)": [height],
            "Weight (kg)": [weight],
            "Daily_total_calory_intake_kcal": [daily_intake],
            "Income Level": [income]
        })
        pred = model.predict(user)[0]
        st.session_state.predicted = pred
        st.session_state.gap = pred - daily_intake
        st.session_state.total_consumed = 0
        st.session_state.option4_list = []
        st.session_state.option5_list = []

# ======================
# OPTIONS
# ======================
if st.session_state.predicted is not None:

    gap = st.session_state.gap - st.session_state.total_consumed
    st.success(f"Predicted requirement: {st.session_state.predicted:.1f} kcal")
    st.info(f"Current gap: {gap:.1f} kcal")

    option = st.radio(
        "Choose option",
        ["1 - Portion size for a given snack", "2 - Suitability of eating a given number of items", "3 - One fixed snack + one suggested snack", "4 - Multiple snacks until energy fulfilled", "5 - Multiple snacks with fixed quantities"]
    )

    food = st.selectbox("Select snack", df_snacks["Name"].unique())
    row = df_snacks[df_snacks["Name"] == food].iloc[0]

    # OPTION 1
    if option == "1":
        portion = gap / row["Energy_kcal_g"]
        items = portion / row["Serving_Size"]
        st.write(f"Items required: {items:.2f}")
        st.write(f"Quantity: {portion:.2f} {row['Size_unit']}")
        food_portion_atlas(row)
        macro_pie_chart(row)

    # OPTION 2
    elif option == "2":
        items = st.number_input("Items eaten", 0.0, 50.0, 1.0)
        energy = items * row["Serving_Size"] * row["Energy_kcal_g"]
        diff = gap - energy
        st.write(f"Energy: {energy:.1f} kcal")
        st.write(f"{'Remaining' if diff>=0 else 'Excess'} gap: {abs(diff):.1f} kcal")
        food_portion_atlas(row)
        macro_pie_chart(row)

    # OPTION 3
    elif option == "3":
        items = st.number_input("Items eaten", 0.0, 50.0, 1.0)
        energy = items * row["Serving_Size"] * row["Energy_kcal_g"]
        new_gap = gap - energy
        st.write(f"Remaining gap: {new_gap:.1f} kcal")
        food_portion_atlas(row)
        macro_pie_chart(row)

        if new_gap > 0:
            food2 = st.selectbox("Second snack", df_snacks["Name"].unique())
            row2 = df_snacks[df_snacks["Name"] == food2].iloc[0]
            portion2 = new_gap / row2["Energy_kcal_g"]
            items2 = portion2 / row2["Serving_Size"]
            st.write(f"Items required: {items2:.2f}")
            st.write(f"Quantity: {portion2:.2f} {row2['Size_unit']}")
            food_portion_atlas(row2)
            macro_pie_chart(row2)

    # OPTION 4
    elif option == "4":
        items = st.number_input("Items eaten", 0.0, 50.0, 1.0)
        if st.button("Add snack"):
            energy = items * row["Serving_Size"] * row["Energy_kcal_g"]
            st.session_state.total_consumed += energy
            st.session_state.option4_list.append(
                (food, items, items*row["Serving_Size"], energy)
            )

        for f, i, q, e in st.session_state.option4_list:
            st.write(f"{f}: {i} items ({q:.1f} {row['Size_unit']}) → {e:.1f} kcal")

        new_gap = st.session_state.gap - st.session_state.total_consumed
        if new_gap > 0:
            st.info(f"Remaining gap: {new_gap:.1f} kcal")
        elif new_gap == 0:
            st.success("Exact amount used")
        else:
            st.error(f"You are using too much: {abs(new_gap):.1f} kcal excess. ")
            st.warning(
              "You have to click on the **'Predict Requirement'** button again "
              "to do another portion prediction using any other option."
            )
    # OPTION 5
    elif option == "5":
        items = st.number_input("Items eaten", 0.0, 50.0, 1.0)
        if st.button("Add snack"):
            energy = items * row["Serving_Size"] * row["Energy_kcal_g"]
            st.session_state.option5_list.append(
                (food, items, energy)
            )

        total = 0
        for f, i, e in st.session_state.option5_list:
            total += e
            st.write(f"{f}: {i} items → {e:.1f} kcal")

        gap5 = st.session_state.gap - total
        st.info(f"Total energy: {total:.1f} kcal")
        st.info(f"Energy gap: {gap5:.1f} kcal")
