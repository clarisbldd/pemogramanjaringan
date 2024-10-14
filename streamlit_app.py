import streamlit as st
import pandas as pd

st.title("Car Evaluation Data")
st.write("Data Overview")

# Load the dataset
df_cars = pd.read_csv("car_evaluation_with.csv", header=None)

# Assign column names for better understanding
df_cars.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Filter the dataset for 'unacc' and 'acc'
df_filtered = df_cars[df_cars['class'].isin(['unacc', 'acc'])]

# Count occurrences of each class
class_counts = df_filtered['class'].value_counts()

# Create a DataFrame for plotting and display
df_class_counts = pd.DataFrame(class_counts).reset_index()
df_class_counts.columns = ['class', 'count']

# Display the counts in a table
st.write("Class Counts Table")
st.dataframe(df_class_counts)

# Display the filtered dataset in a table
st.write("Filtered Dataset")
st.dataframe(df_filtered)

#
# Area Chart
#
st.title("Bar Chart of Car Classes")
st.bar_chart(df_class_counts.set_index('class'))

st.markdown("---")
