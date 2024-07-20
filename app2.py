import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = 'C:\\Users\\shrijal\\Desktop\\Kdrama_hit_flop_predictor\\ML_MODEL\\logistic_regression_model_blabla.pkl'
model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    # Set the title of the web app
    st.title('K-Drama Hit/Flop Prediction')

    # Add a description
    st.write('Enter Information')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('K-Drama Information')

        # Add input fields for features
        name = st.text_input('Drama Name')
        
        story_score = st.slider("Story Score", 0.0, 10.0, 0.0, 0.1)
        acting_cast_score = st.slider("Acting Score", 0.0, 10.0, 0.0, 0.1)
        music_score = st.slider("Music Score", 0.0, 10.0, 0.0, 0.1)
        rewatch_value_score = st.slider("Rewatch Score", 0.0, 10.0, 0.0, 0.1)
        n_helpful = st.slider("Not Helpful Score", 0, 30, 0, 1)
        overall_score = st.slider("Overall Score", 0.0, 10.0, 0.0, 0.1)
        
        untrained_column = st.text_input('Additional Information (not used in prediction)')
        
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'story_score': [story_score],
        'acting_cast_score': [acting_cast_score],
        'music_score': [music_score],
        'rewatch_value_score': [rewatch_value_score],
        'overall_score': [0.0],
        'n_helpful': [n_helpful]
    })
    
    # Ensure columns are in the same order as during model training
    try:
        input_data = input_data[expected_columns]
    except KeyError as e:
        st.error(f"Error: {e}")
        st.stop()
    
    # Log input data for debugging
    st.write("Input Data:")
    st.write(input_data)

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            try:
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.stop()
            
            st.write(f'Prediction for {name}: {"Hit" if prediction[0] == 1 else "Flop"}')
            st.write(f'Probability of Being a Hit: {probability:.2f}')
            #st.write(f'Overall Score: {overall_score}')

            # Plotting
            fig, axes = plt.subplots(3, 1, figsize=(8, 16))

            # Plot Hit/Fail probability
            sns.barplot(x=['Flop', 'Hit'], y=[1 - probability, probability], ax=axes[0], palette=['red', 'green'])
            axes[0].set_title('Flop/Hit Probability')
            axes[0].set_ylabel('Probability')

            # Plot GPA distribution
            sns.histplot(input_data['overall_score'], kde=True, ax=axes[1])
            axes[1].set_title('Overall Distribution')

            # Plot Pass/Fail pie chart
            axes[2].pie([1 - probability, probability], labels=['Flop', 'Hit'], autopct='%1.1f%%', colors=['red', 'green'])
            axes[2].set_title('Hit/Flop Pie Chart')

            # Display the plots
            st.pyplot(fig)

            # Provide recommendations
            if prediction[0] == 1:
                st.success(f"{name} is likely to be a hit. Keep up the good work!")
            else:
                st.error(f"{name} is likely to flop. Consider improving story and Actor's acting.")

if __name__ == '__main__':
    main()
