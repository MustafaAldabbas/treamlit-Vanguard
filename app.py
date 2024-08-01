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
    
    raw_data_tabs = st.tabs(["Demographic Data", "Old Website Data", "New Website Data", "Experiment Data", "Clean Data"])
    
    with raw_data_tabs[0]:
        st.write("##### Demographic Data")
        st.write(" The dataset contains demographic data of the clients who are part of the experience, like age, tenure, gender, balance")
        st.dataframe(df_demo.head())
    
    with raw_data_tabs[1]:
        st.write("##### Old Website Data")
        st.write("This data is retrieved from the old website of Vanguard and includes information about visits and time spent on the website")
        st.dataframe(df_web_data1.head())
    
    with raw_data_tabs[2]:
        st.write("##### New Website Data")
        st.write("This data is retrieved from the new website of Vanguard and includes information about visits and time spent on the website")
        st.dataframe(df_web_data2.head())
    
    with raw_data_tabs[3]:
        st.write("##### Experiment Data")
        st.write("This is the data of the experiment, which includes a unique ID for each client and to which group they belong, the test or the control group")
        st.dataframe(df_experiment_clients.head())
    
    with raw_data_tabs[4]:
        st.write("### Clean Data")
        st.write("We applied general cleaning methods on the 4 datasets, merged them based on the client ID, and cleaned null values")
        st.dataframe(df.head())
    
    eda_tabs = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Experiment Sample Characteristics"])
    
    with eda_tabs[0]:
        st.write("## EDA Univariate Analysis")
        st.write("**1-Distribution of Age**")
        st.write("**2-Distribution of Tenure**")
        st.write("**3-Distribution of Balance**")
        st.write(" **4-Distribution of Logins in the last 6 months**")
        st.write(" **5-Distribution of Calls in the last 6 months**")
        fig, results = fn.UnivariateAnalysis_st(df)
        st.pyplot(fig)
    
    with eda_tabs[1]:
        st.write("## EDA Bivariate Analysis")
        st.write("**1-Noticeable differences in balances across genders.**")
        st.write("**2-Clients who have more accounts have more balance.**")
        fig, results = fn.BivariateAnalysisst(df)
        st.pyplot(fig)
    
    with eda_tabs[2]:
        st.write("## Experiment Sample Characteristics")
        st.write("**Study clients counts**: Total of 50487. ")
        st.write("**Gender Distribution**: Balanced gender representation. ")
        st.write("**Age Distribution**: Predominantly between 30-60 years old, with range from 20 - 90.")
        fig, gender_distribution, client_group_counts, age_dist_test, age_dist_control = fn.combined_analysis_st(df)
        st.pyplot(fig)

def hypothesis_testing_page():
    st.title("Hypothesis Testing 🧪")
    
    st.write("## Key Performance Metrics")
    st.write("* Completion Rate: Percentage of clients completing the process.")
    st.write("* Average time per step: Time Spent on Each Step.")
    st.write("* Error Rates: The Frequency of errors.")
    
    hypotheses = [
        "Hypothesis 1",
        "Hypothesis 2",
        "Hypothesis 3",
        "Hypothesis 4",
        "Hypothesis 5"
    ]
    
    hypothesis_tabs = st.tabs(hypotheses)
    
    with hypothesis_tabs[0]:
        st.write("### The new Website increases completion rates")
        st.write("* Higher completion rate in the test group.")
        st.write("* The Two-proportion z-test (9.08) confirms significance difference. Extremely low p_value indicated the difference is statistically significant")
        st.write("* Conclusion: The new UI drives more confirmations.")
        fig, results = fn.analyze_experiment_completion_ratesst(df)
        st.pyplot(fig)
    
    with hypothesis_tabs[1]:
        st.write("### Males confirm more than females")
        st.write("* Gender Analysis: Males confirm more often in both groups.")
        st.write("* Conclusion: Confirmation rates are higher for males.")
        fig, summary_test, summary_control, chi_square_results = fn.analyze_confirmation_by_genderst(df)
        st.pyplot(fig)
    
    with hypothesis_tabs[2]:
        st.write("### The new Website reduces time spent on steps")
        st.write("* Time Spent: Less time spent on Start, step1, and confirm in the test group.")
        fig, avg_time_df, t_test_df = fn.analyze_time_per_step_between_groupsst(df)
        st.pyplot(fig)
    
    with hypothesis_tabs[3]:
        st.write("### The new Website decreases error rates")
        st.write("##### We Identified Error as Jumping between steps back and forth")
        st.write("* Error Rate is Lower in control group.")
        st.write("* Actionable Insight: Identify and address the root causes of the increased error rate. This could involve usability testing, bug fixing, or user feedback to improve the website.")
        fig, results = fn.analyze_experiment_error_ratesst(df)
        st.pyplot(fig)
    
    with hypothesis_tabs[4]:
        st.write("### The new UI increases client’s login frequency")
        st.write("* Login Frequency: Similar in both groups")
        st.write("* Conclusion: The new UI Has no negative impact on login frequency")
        fig, results_df, additional_stats = fn.analyze_logins_between_groupsst(df)
        st.pyplot(fig)
    
    

def recommendations_page():
    st.title("Recommendations and Conclusion 📝")
    
    st.subheader('Recommendations') 
    st.write("* Refine UI: Based on client feedback and additional data.")
    st.write("* Conduct Further Experiments: Testing new features and improvements.")
    st.write("* Enhance Data Collection: Gather more comprehensive insights.")
    st.write("* Error Testing: Check step 2 and step 3 to understand why errors have increased.")
    
    st.subheader('Conclusion')
    st.write("* The new UI significantly improves user experience based on completion rates. However, the error rate in the new UI is higher.")
    st.write("* Recommendations: Continue refining the UI, conduct further experiments, and enhance data collection.")

def download_report():
    # Implementation of report generation
    output = BytesIO()
    # Generate report content here
    return output

def main():
    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", "EDA", "Hypothesis Testing", "Recommendations and Conclusion"],
                               icons=['house', 'bar-chart', 'clipboard-data', 'file-text'], menu_icon="cast", default_index=0)

    if selected == "Home":
        home_page()
    elif selected == "EDA":
        eda_page()
    elif selected == "Hypothesis Testing":
        hypothesis_testing_page()
    elif selected == "Recommendations and Conclusion":
        recommendations_page()

    st.markdown("---")
    st.markdown("Created by Dan Sun and Mustafa Aldabbas")

    if st.button("Download Full Report"):
        output = download_report()
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="vanguard_analysis_report.pdf">Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
