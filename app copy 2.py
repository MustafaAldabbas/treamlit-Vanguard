import streamlit as st
import pandas as pd
import functions as fn
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
from streamlit_option_menu import option_menu
import plotly.express as px
import base64
from io import BytesIO
from streamlit_option_menu import option_menu

# Set pandas display options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)

# Read the data files
df_demo = pd.read_csv(r'/Users/mustafaaldabbas/Documents/GitHub/Project_streamlit/Raw Data/df_final_demo (1).txt', delimiter=',', header=0)
df_web_data1 = pd.read_csv(r'/Users/mustafaaldabbas/Documents/GitHub/Project_streamlit/Raw Data/df_final_web_data_pt_1.txt', delimiter=',', header=0)
df_web_data2 = pd.read_csv(r'/Users/mustafaaldabbas/Documents/GitHub/Project_streamlit/Raw data/df_final_web_data_pt_2.txt', delimiter=',', header=0)
df_experiment_clients = pd.read_csv(r'/Users/mustafaaldabbas/Documents/GitHub/Project_streamlit/Raw data/df_final_experiment_clients.txt', delimiter=',', header=0)

# Merge and clean data
df = fn.merge_clean_transform_data(df_web_data1, df_web_data2, df_demo, df_experiment_clients, on='client_id', n=5)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    color: #0066cc;
}
.stButton>button {
    color: #ffffff;
    background-color: #0066cc;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

def home_page():
    st.title('Vanguard experiment A/B testing 🚀')
    st.markdown('<p class="big-font">Uncover the Power of UI Improvements!</p>', unsafe_allow_html=True)
    
    st.write("## Introduction")
    st.info("As a newly employed data analyst in the Customer Experience (CX) team at Vanguard, you've been tasked with analyzing the results of an exciting digital experiment.")
    
    st.write("### The Digital Challenge")
    st.write("Vanguard believed that a more intuitive and modern User Interface (UI), coupled with timely in-context prompts, could make the online process smoother for clients. The critical question was: Would these changes encourage more clients to complete the process?")
    
    st.write("### The Experiment Conducted")
    st.write("An A/B test was conducted from 3/15/2017 to 6/20/2017:")
    st.write("- Control Group: Clients interacted with Vanguard's traditional online process.")
    st.write("- Test Group: Clients experienced the new, spruced-up digital interface.")
    
    st.write("### Our Tools")
    st.write("We're working with three main datasets:")
    st.write("1. Client Profiles (df_final_demo)")
    st.write("2. Digital Footprints (df_final_web_data)")
    st.write("3. Experiment Roster (df_final_experiment_clients)")

def eda_page():
    st.title("Exploratory Data Analysis 📊")
    
    st.write("## Raw Data Overview")
    
    tabs = st.tabs(["Demographic Data", "Old Website Data", "New Website Data", "Experiment Data", "Clean Data"])
    
    with tabs[0]:
        st.write("##### Demographic Data")
        st.dataframe(df_demo.head())
    
    with tabs[1]:
        st.write("##### Old Website Data")
        st.dataframe(df_web_data1.head())
    
    with tabs[2]:
        st.write("##### New Website Data")
        st.dataframe(df_web_data2.head())
    
    with tabs[3]:
        st.write("##### Experiment Data")
        st.dataframe(df_experiment_clients.head())
    
    with tabs[4]:
        st.write("### Clean Data")
        st.write("We applied general cleaning methods on the 4 datasets, merged them based on the client id, and cleaned null values.")
        st.dataframe(df.head())
    
    st.write("## EDA Univariate Analysis")
    fig, results = fn.UnivariateAnalysis_st(df)
    st.pyplot(fig)
    
    st.write("## EDA Bivariate Analysis")
    fig, results = fn.BivariateAnalysisst(df)
    st.pyplot(fig)
    
    st.write("## Experiment Sample Characteristics")
    fig, gender_distribution, client_group_counts, age_dist_test, age_dist_control = fn.combined_analysis_st(df)
    st.pyplot(fig)

def hypothesis_testing_page():
    st.title("Hypothesis Testing 🧪")
    
    st.write("## Key Performance Metrics")
    st.write("* Completion Rate: Percentage of clients completing the process.")
    st.write("* Average time per step: Time Spent on Each Step.")
    st.write("* Error Rates: The Frequency of errors.")
    
    hypotheses = [
        "New Website Increases Completion Rates",
        "Males Confirm More Than Females",
        "New Website Reduces Time Spent on Steps",
        "New Website Decreases Error Rates",
        "New UI Increases Client's Login Frequency"
    ]
    
    selected_hypothesis = st.selectbox("Choose a hypothesis to explore:", hypotheses)
    
    if selected_hypothesis == "New Website Increases Completion Rates":
        st.write("### The new Website increases completion rates")
        fig, results = fn.analyze_experiment_completion_ratesst(df)
        st.pyplot(fig)
    
    elif selected_hypothesis == "Males Confirm More Than Females":
        st.write("### Males confirm more than females")
        fig, summary_test, summary_control, chi_square_results = fn.analyze_confirmation_by_genderst(df)
        st.pyplot(fig)
    
    elif selected_hypothesis == "New Website Reduces Time Spent on Steps":
        st.write("### The new Website reduces time spent on steps")
        fig, avg_time_df, t_test_df = fn.analyze_time_per_step_between_groupsst(df)
        st.pyplot(fig)
    
    elif selected_hypothesis == "New Website Decreases Error Rates":
        st.write("### The new Website decreases error rates")
        fig, results = fn.analyze_experiment_error_ratesst(df)
        st.pyplot(fig)
    
    elif selected_hypothesis == "New UI Increases Client's Login Frequency":
        st.write("### The new UI increases client's login frequency")
        fig, results_df, additional_stats = fn.analyze_logins_between_groupsst(df)
        st.pyplot(fig)

def custom_analysis_page():
    st.title("Custom Analysis 🔍")
    
    st.write("Upload your own dataset for custom analysis.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df_custom = pd.read_csv(uploaded_file)
        st.write("Preview of your data:")
        st.dataframe(df_custom.head())
        
        st.write("Select columns for analysis:")
        selected_columns = st.multiselect("Choose columns", df_custom.columns)
        
        if selected_columns:
            st.write("Basic statistics:")
            st.write(df_custom[selected_columns].describe())
            
            st.write("Correlation heatmap:")
            fig, ax = plt.subplots()
            sns.heatmap(df_custom[selected_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

def download_report():
    output = BytesIO()
    # Generate report content here (you'll need to implement this)
    return output

def main():
    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", "EDA", "Hypothesis Testing", "Custom Analysis"],
                               icons=['house', 'bar-chart', 'clipboard-data', 'gear'], menu_icon="cast", default_index=0)

    if selected == "Home":
        home_page()
    elif selected == "EDA":
        eda_page()
    elif selected == "Hypothesis Testing":
        hypothesis_testing_page()
    elif selected == "Custom Analysis":
        custom_analysis_page()

    st.markdown("---")
    st.markdown("Created by Dan Sun and Mustafa Aldabbas")

    if st.button("Download Full Report"):
        output = download_report()
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="vanguard_analysis_report.pdf">Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == '__main__':
    main()