


# app.py

import streamlit as st


import pandas as pd
import functions as fn
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import emoji

# Read the text file into a DataFrame
df_demo = pd.read_csv(r'/Users/mustafaaldabbas/Documents/GitHub/Project_streamlit/Raw Data/df_final_demo (1).txt', delimiter=',', header=0)
df_web_data1 = pd.read_csv(r'/Users/mustafaaldabbas/Documents/GitHub/Project_streamlit/Raw Data/df_final_web_data_pt_1.txt', delimiter=',', header=0)
df_web_data2 = pd.read_csv(r'/Users/mustafaaldabbas/Documents/GitHub/Project_streamlit/Raw data/df_final_web_data_pt_2.txt', delimiter=',', header=0)
df_experiment_clients = pd.read_csv(r'/Users/mustafaaldabbas/Documents/GitHub/Project_streamlit/Raw data/df_final_experiment_clients.txt', delimiter=',', header=0)

df=fn.merge_clean_transform_data(df_web_data1, df_web_data2, df_demo, df_experiment_clients, on='client_id', n=5)


def main():
    st.title('Vanguard experiment A/B testing ')
    st.write("## Introduction ")
    st.write("##### Context ")
    st.write("I'm newly employed data analyst in the Customer Experience (CX) team at Vanguard, the US-based investment management company. You’ve been thrown straight into the deep end with your first task. Before your arrival, the team launched an exciting digital experiment, and now, they’re eagerly waiting to uncover the results and need your help!")
    st.write("##### The Digital Challenge ")
    st.write("Vanguard believed that a more intuitive and modern User Interface (UI), coupled with timely in-context prompts (cues, messages, hints, or instructions provided to users directly within the context of their current task or action), could make the online process smoother for clients. The critical question was: Would these changes encourage more clients to complete the process?")
    st.write("##### The Experiment Conducted ")
    st.write("An A/B test was set into motion from 3/15/2017 to 6/20/2017 by the team.Control Group: Clients interacted with Vanguard’s traditional online process.Test Group: Clients experienced the new, spruced-up digital interface.Both groups navigated through an identical process sequence: an initial page, three subsequent steps, and finally, a confirmation page signaling process completion.The goal is to see if the new design leads to a better user experience and higher process completion rates ")
    st.write("##### My Tools")
    st.write("There are three datasets that will guide your journey:Client Profiles (df_final_demo): Demographics like age, gender, and account details of our clients.Digital Footprints (df_final_web_data): A detailed trace of client interactions online, divided into two parts: pt_1 and pt_2. It’s recommended to merge these two files prior to a comprehensive data analysis.Experiment Roster (df_final_experiment_clients): A list revealing which clients were part of the grand experiment.")


    # Interactive widgets
    st.sidebar.header('Controls')
    min_rating = st.sidebar.slider('Minimum Rating', min_value=0, max_value=10, value=5, step=1)
    



# Raw Data 
    st.write("### Raw Data")
    st.write("##### **Demographic Data**")
    st.write(" The dataset contain a demographic data of the clients who are part of the expierience like age, tenure, gender, balance")
    st.dataframe(df_demo.head(5))

    st.write("##### **Old Website Data**")
    st.write("This data is retrieved from the website old website of vanguard and include information about visits and time spent on the website")
    st.dataframe(df_web_data1.head(5))

  
   
    st.write("##### **New Website Data**")
    st.write("This data is retrieved from the new website of vanguard and include information about visits and time spent on the website")
    st.dataframe(df_web_data2.head(5))
    

    st.write("##### **Experiment Data**")
    st.write("This is the Data of the experiment which include a unique ID for each client and to which group they belon, the test or the control group")
    st.dataframe(df_experiment_clients.head(5))
    



   #discription of the Clean data
    # Clean Data set 
    st.write("### Clean Data")
    st.write("We applied General cleaning methodes on the 4 datasets then we merged them based on the client id, made sure to change datatypes and clean null values")
    st.dataframe(df.head(5))
    
     
    st.write("# Is the new website better?")
    st.write(" To answer this questions we need do do the Analysis, but to To better understand our data and the clients, we started by doing an EDA Analysis")
    #EDA Analysis 
    st.write("## EDA Univariante Analysis")
    st.write("**1-Distribution of Age**")
    st.write("**2-Distribution of Tenure**")
    st.write("**3-Distribution of Balance**")
    st.write(" **4-Distribution of Logins in the last 6 months**")
    st.write(" **5-Distribution of Calls in the last 6 months**")

    fig, results = fn.UnivariateAnalysis_st(df)
    st.pyplot(fig)

    
    st.write("## EDA Bivariante Analysis")
    st.write("**1-Noticeable differences in balances across genders.**")
    st.write("**2-Clients who have more accounts have more balance.**")
    fig, results = fn.BivariateAnalysisst(df)
    st.pyplot(fig)
 



       #Study Test and control group 
    st.write("## Experiment Sample Characteristics")
    st.write("**Study clients counts**: Total of 50487. ")
    st.write("**Gender Distribution**: Balanced gender representation. ")
    st.write("**Age Distribution**: Predominantly between 30-60 years old, with range from 20 - 90.")


    fig, gender_distribution, client_group_counts, age_dist_test, age_dist_control = fn.combined_analysis_st(df)
    st.pyplot(fig)
    


    #Key Performance Metrics
    st.subheader('Key Performance Metrics') 
    st.write("* Completion Rate: Percentage of clients completing the process.")
    st.write("* Average time per step: Time Spent on Each Step.")
  
    st.write("* Error Rates: The Frequency of errors.")
   

    #Hypothesis
    st.subheader('Hypothesis') 
    #Hypothesis 1
    st.write("### The new Website increases completion rates")
    st.write("* Higher completion rate in the test group.")
    st.write("* The Two-proportion z-test (9.08) confirms significance difference. Extremely low p_value indicated the difference is statistically significant")
    st.write("* Conclusion: The new UI drives more confirmations. ")
    fig, results = fn.analyze_experiment_completion_ratesst(df)
    st.pyplot(fig)


    # Hypothesis 2
    st.write("### Males confirm more than females")
    st.write("* Gender Analysis: Males confirm more often in both groups. ")
    st.write("* Conclusion: Confirmation rates are higher for males. ")
    fig, summary_test, summary_control, chi_square_results = fn.analyze_confirmation_by_genderst(df)
    st.pyplot(fig)
    

    # Hypothesis 3
    st.write("### The new Website reduces time spent on steps")
    st.write("* Time Spent: Less time spent on  Start, step1 and confirm  in the test group. ")
    fig, avg_time_df, t_test_df = fn.analyze_time_per_step_between_groupsst(df)
    st.pyplot(fig)


    #Hypothesis 4
    st.write("###The new Website decreases error rates")
    st.write("##### We Identified Error as Jumping between steps  back and forth")
    st.write("* Error Rate is Lower in control group.")
    st.write("* Actionable Insight Identify and address the root causes of the increased error rate. This could involve usability testing, bug fixing, or user feedback to improve the website.")
    fig, results = fn.analyze_experiment_error_ratesst(df)
    st.pyplot(fig)

    # Hypothesis 5
    st.write("### The new UI increases client’s login frequency ")
    st.write("* Login Frequency: Similar in both groups")
    st.write("* Conclusion: The new UI Has no negative impact on login frequency")
    fig, results_df, additional_stats = fn.analyze_logins_between_groupsst(df)
    st.pyplot(fig)

    
    
    st.subheader('Hypothesis') 
    st.write("* The new UI significantly improves user experience based on completion rates, However, new UI error rate is higher. ")
    st.write("* Recommendations: Continue refining the UI, conduct further experiments, and enhance data collection.")

    st.subheader('Recommendations') 
    st.write("* Refine UI: Based on client feedback and additional data ")
    st.write("* Conduct Further Experiments: Testing new features and improvements.")
    st.write("* Enhance Data Collection: Gather more comprehensive insights ")
    st.write("* Error Testing: checking step2 and step 3 why the errors are increased")
    
    st.subheader('Auther') 
    st.write("* Dan Sun")
    st.write("* Mustafa Aldabbas ")

   

    # Plotting
    #st.write("### Sales Over Time")
    #plt = plot_sales_over_time(filtered_data)
    #st.pyplot(plt)
    


if __name__ == '__main__':
    main()
