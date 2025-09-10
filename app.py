# Before - checkbox page for model
import base64
from sklearn.tree import DecisionTreeClassifier
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score
# 52 to 55 - uncomment (st.session - if block)
# 37 - uncomment - csv input

# #--------- Variable

csv_file = None

if "target" not in st.session_state:
    st.session_state["target"] = "None"

if "model" not in st.session_state:
    st.session_state["model"] = list()

if "columns_info" not in st.session_state:
    st.session_state["columns_info"] = "None"

if "df" not in st.session_state:
    st.session_state["df"] = "None"

if "cat_col_names2" not in st.session_state:
    st.session_state["cat_col_names2"] = "None"

if "filtered_df" not in st.session_state:
    st.session_state["filtered_df"] = "None"

if "filtered_df2" not in st.session_state:
    st.session_state["filtered_df2"] = "None"

#------------- The below variables are contains encoded data and their mappings
if "encoded_df" not in st.session_state:
    st.session_state["encoded_df"] = "None"

if "mappings" not in st.session_state:
    st.session_state["mappings"] = "None"

if "final_df_dep_col_all" not in st.session_state:
    st.session_state["final_df_dep_col_all"] = "None"

if "scaled_df" not in st.session_state:
    st.session_state["scaled_df"] = "None"

if "scaler" not in st.session_state:
    st.session_state["scaler"] = "MinMaxScaler"

if "results" not in st.session_state:
    st.session_state["results"] = []
if "model_files" not in st.session_state:
    st.session_state["model_files"] = {}

# Initialize session state for storing results
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "download_links" not in st.session_state:
    st.session_state.download_links = {}


if "scaler_file" not in st.session_state:
    st.session_state.scaler_file = None


#--- Train and test data -----
if "x_train" not in st.session_state:
    st.session_state["x_train"] = "None"
if "x_test" not in st.session_state:
    st.session_state["x_test"] = "None"
if "y_train" not in st.session_state:
    st.session_state["y_train"] = "None"
if "y_test" not in st.session_state:
    st.session_state["y_test"] = "None"
#-------


#------------- 



#-------------


#--------------

# Set page configuration
st.set_page_config(
    layout="wide"  # Enable wide mode
)

# Define a function to manage navigation across pages
def navigate(page):
    st.session_state["current_page"] = page

# Initialize session state for page navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

# page 2
def page2():
    
    global df

    st.button("Go Back", on_click=navigate, args=("Home",))
    st.title("Upload a file to proceed", anchor=False)
    csv_file = st.file_uploader("Upload your file", type=["csv"])
    # csv_file = "heart.csv" 
    
    if csv_file is not None:
        
        st.session_state["uploaded_file"] = csv_file
        
        df = pd.read_csv(csv_file)
        st.write(df.head(5))
        columns_list = df.columns.tolist()

        # Get the number of columns
        num_columns = len(columns_list)

        # Output the list and number of columns
        st.write(f"Number of columns: {num_columns}")
        num_rows = df.shape[0]
        # Output the number of rows
        st.write(f"Number of rows: {num_rows}")
        
        st.session_state["df"] = df
        st.button("Submit", on_click=navigate, args=("page 3",))  # Use on_click
    st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        color: #333;
    }
    </style>
    <div class="footer">
        Developed by Manindra
    </div>
    """,
    unsafe_allow_html=True,
)
        


def is_categorical(column, threshold=4):
    """
    Determine if a column is categorical based on its unique value count.
    
    Parameters:
    column (pd.Series): The column to check.
    threshold (int): The maximum number of unique values to consider as categorical.
    
    Returns:
    bool: True if the column is categorical, otherwise False.
    """
    unique_values = column.nunique()  # Count of unique values
    return unique_values <= threshold  # Compare with threshold


def is_string_column(df, column):
    return df[column].dtype == 'object'

from pandas.api.types import is_string_dtype


def get_readable_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "Integer"
    elif pd.api.types.is_float_dtype(dtype):
        return "Float"
    elif pd.api.types.is_string_dtype(dtype):
        return "Text"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "Date/Time"
    else:
        return "Other"

# page 3
def page3():
    st.button("Go Back", on_click=navigate, args=("page 2",))  # Use on_click
    st.title("Feature selection",anchor=False)
    df = st.session_state["df"]
    # st.write(df.columns)
    columns_info = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': [get_readable_dtype(df[col].dtype) for col in df.columns],
    'Unique Values': [df[col].nunique() for col in df.columns]
    })
    columns_info['Column Name'] =  columns_info['Column Name'].astype(str)
    columns_info['Data Type'] =  columns_info['Data Type'].astype(str)
    columns_info['Unique Values'] =  columns_info['Unique Values'].astype(str)

    text_columns = columns_info[columns_info["Data Type"] == "Text"]["Column Name"].tolist()

    columns_to_lower = text_columns
    dfx = df.copy()
    dfx[columns_to_lower] = dfx[columns_to_lower].apply(lambda x: x.str.lower())
    df = dfx.copy()

    # st.write(text_columns)
    # st.write(columns_info)
    rotated_df = columns_info.transpose()
    st.session_state['columns_info'] = columns_info.copy()

    columns_info = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': [get_readable_dtype(df[col].dtype) for col in df.columns],
    'Unique Values': [df[col].nunique() for col in df.columns]
    })
    columns_info['Column Name'] =  columns_info['Column Name'].astype(str)
    columns_info['Data Type'] =  columns_info['Data Type'].astype(str)
    columns_info['Unique Values'] =  columns_info['Unique Values'].astype(str)

    rotated_df = columns_info.transpose()
    # Display the rotated DataFrame
    st.write(rotated_df)



    
    # Display the uploaded file details
    # if st.session_state["uploaded_file"]:
    #     st.write("File name:", st.session_state["uploaded_file"].name)
    # else:
    #     st.write("No file uploaded.")
    
    thres_hold = st.number_input("Enter Threshold (to classify categorical values) : ",value=4,step=1,min_value=2)
    st.session_state['final_df_dep_col_all'] = [col for col in df.columns if(is_categorical(df[col], thres_hold) or not is_string_dtype(df[col]))]
    

    st.write("Unsupported colunms, because they are not categorical but textual columns")
    # if(st.session_state['final_df_dep_col_all']!= "None"):
    list_b = st.session_state['final_df_dep_col_all']
    all_columns = df.columns.tolist()
    # st.write(all_columns)

    list_to_show = [item for item in all_columns if item not in list_b]
    
    for x in list_to_show:
        st.error(x)
    

    col1, col2 = st.columns(2)
    with col1:
        all_columns = [col for col in df.columns if(is_categorical(df[col], thres_hold) or not is_string_dtype(df[col]))]
        categorical_columns = [col for col in df.columns if ( is_categorical(df[col], thres_hold))]
        non_categorical_columns = [col for col in df.columns if  ( not is_categorical(df[col],thres_hold))]
        all_col_names = df[all_columns]
        non_cat_col_names =  df[non_categorical_columns]
        cat_col_names =  df[categorical_columns]
        st.session_state.cat_col_names2 = cat_col_names.columns
        final_df_indep_cols = []
        # final_df_indep_cols.extend([True]*100)
        final_df_dep_col = ""
        # col1, col2 = st.columns([1,1])
        
        st.header("Select independent columns, including dependent also (choose atleast two) ", anchor=False)
        n = all_col_names.shape[1]
        s_all = st.radio("Choose",["Select all", "Deselect all"])
            
        st.session_state.toggs_arr = []
        st.session_state.toggs_arr.extend([True]*n)
            

        if(s_all == "Select all"):
            for i in range(0,n):
                st.session_state.toggs_arr[i] = True

        if(s_all == "Deselect all"):
            for i in range(0,n):
                st.session_state.toggs_arr[i] = False
        

        for i in range(0,n):
            # st.write(non_cat_col_names.columns[i])
            st.session_state.toggs_arr[i] = st.toggle(""+all_col_names.columns[i], value=st.session_state.toggs_arr[i])
            # now you have the index of the independent columns index
        for i in range(0,n):
            if(st.session_state.toggs_arr[i] == True):
                # st.write(non_cat_col_names.columns[0])
                final_df_indep_cols.append(all_col_names.columns[i])
        # st.write(final_df_indep_cols)
        
        st.session_state['final_df_dep_col_all'] = df[final_df_indep_cols].copy()
    
    # create a radio button
    # The choice selected will make the independent choice to false

#-------
    with col2:
        st.header("Dependent columns(choose one)",anchor=False)
        cat_col_names = [col for col in final_df_indep_cols if (is_categorical(df[col], thres_hold))]
        n = len(cat_col_names)

        # create dependent columns list from the 
        
        final_df_dep_col = st.radio("Select Dependent column ", cat_col_names,0)
        # st.write(final_df_dep_col)
        st.session_state["target"] = final_df_dep_col
        #when element selected then make the independents toggle to false
        # all_col_names.remove(final_df_dep_col)
        temp = all_col_names.shape[1]
        all_col_names2 =[] 
        for i in range(0,temp):
            all_col_names2.append(all_col_names.columns[i])
        if(final_df_dep_col):
            ind = all_col_names2.index(final_df_dep_col)
            st.session_state.toggs_arr[ind] = False
            final_df_indep_cols.remove(final_df_dep_col)
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    # st.button("Submit attributes", on_click=toggle_action(final_df_indep_cols,final_df_dep_col))
    if True:
        if(final_df_dep_col == None):
            st.warning("Invalid number of attributes, Atleast ondependent column and one independent column necessary")
            model_creation = False
        elif(len(final_df_indep_cols) == 0):
            st.warning("Invalid number of attributes, The independent and dependent should be different")
            model_creation = False
        else:
            # st.write(final_df_indep_cols)
            # st.write(final_df_dep_col)
            model_creation = True
            st.button("Submit", on_click=navigate, args=("page 4",))  # Use on_click
    st.markdown('</div>', unsafe_allow_html=True)
    # filtered_df = st.session_state["filtered_df"]
    try:
        filtered_df = pd.concat( [ df[final_df_indep_cols], df[final_df_dep_col] ], axis=1)
        st.session_state["filtered_df"] = filtered_df

    except:
        st.error("Invalid data, given")

import warnings
warnings.filterwarnings("ignore")
def page4():
    st.button("Go Back", on_click=navigate, args=("page 3",))  # Use on_click
    st.title("Handling NA values",anchor=False)
    
    
    filtered_df = st.session_state["filtered_df"]
    x = filtered_df.copy()
    # st.write(x.head(5))
    # x[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","Outcome"]] = x[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","Outcome"]].replace(0,np.nan) ---Flag

    null_values = pd.DataFrame({
    'Column Name': x.columns,
    'Null Values': x.isnull().sum()})

    null_values = x.isnull().sum()    
    columns_with_nulls = null_values[null_values > 0]
    num_rows = len(columns_with_nulls)
    st.header("Columns containing NaN or null values : ",anchor=False)
    if(num_rows==0):
        st.success("No NA values")
        st.session_state["filtered_df2"] = filtered_df
        sub_col = st.columns(7)
        with sub_col[3]:
            st.button("Submit", on_click=navigate, args=("page 5",))
        return
    else:
        st.dataframe(columns_with_nulls,width=500)
    
    ##############################
    
    columns_with_na = [col for col in filtered_df.columns if filtered_df[col].isnull().any() or (filtered_df[col] == np.nan).any()]
    columns_with_na = [col for col in x.columns if x[col].isnull().any() or (x[col] == np.nan).any()]
    selected_options = []
    # Display the result
    he_cols = st.columns([3,1])
    if(len(columns_with_na) == 0):
        st.write("No columns with Nan or null values")
    else:
        # st.write(columns_with_na)
        elements = columns_with_na
        he_cols[0].title("Choose a Statistical Measures",anchor=False)

        # Options for the radio buttons
        options = ["Mean", "Median", "Mode"]
        selected_options = []


        ##-------
        i = 0
        ui_col = st.columns(4)
        # with ui_col[3]:
        he_cols[1].title("Choosed values ",anchor=False)
        while i < len(elements):
            
            with ui_col[0]:
                if(i==len(elements)):
                    break

                element = elements[i]
                
                if(element in st.session_state.cat_col_names2):
                    st.write("This is categorical")

                    selected_option = st.radio(
                    f"Select for {element}",
                    ["Mode"],
                    key=f"radio_{i}",  # Unique key combining the element index
                    index=0  # Default selection (optional)
                )
                # Create radio buttons only once for each element (horizontally)
                else:
                    selected_option = st.radio(
                        f"Select for {element}",
                        options,
                        key=f"radio_{i}",  # Unique key combining the element index
                        index=0  # Default selection (optional)
                    )
                selected_options.append(selected_option)
                
                # Show the selected option for the current element
                # st.write(f"Selected: {selected_option}")
                st.write("---")
                i += 1
            with ui_col[1]:
                if(i==len(elements)):
                    break
                element = elements[i]
                if(element in st.session_state.cat_col_names2):
                    st.markdown("This is categorical")

                    selected_option = st.radio(
                    f"Select for {element}",
                    ["Mode"],
                    key=f"radio_{i}",  # Unique key combining the element index
                    index=0  # Default selection (optional)
                )
                # Create radio buttons only once for each element (horizontally)
                else:
                    selected_option = st.radio(
                        f"Select for {element}",
                        options,
                        key=f"radio_{i}",  # Unique key combining the element index
                        index=0  # Default selection (optional)
                    )
                selected_options.append(selected_option)
                
                # Show the selected option for the current element
                # st.write(f"Selected: {selected_option}")
                st.write("---")
                i += 1
            with ui_col[2]:
                if(i==len(elements)):
                    break
                element = elements[i]
                if(element in st.session_state.cat_col_names2):
                    st.write("This is categorical")

                    selected_option = st.radio(
                    f"Select for {element}",
                    ["Mode"],
                    key=f"radio_{i}",  # Unique key combining the element index
                    index=0  # Default selection (optional)
                )
                # Create radio buttons only once for each element (horizontally)
                else:
                    selected_option = st.radio(
                        f"Select for {element}",
                        options,
                        key=f"radio_{i}",  # Unique key combining the element index
                        index=0  # Default selection (optional)
                    )
                selected_options.append(selected_option)
                
                # Show the selected option for the current element
                # st.write(f"Selected: {selected_option}")
                st.write("---")
                i += 1

        if(len(columns_with_na)>0):
        # last_df = pd.DataFrame([columns_with_na,selected_options])
            with ui_col[3]:
                last_df = pd.DataFrame({'Columns with NA': columns_with_na,'Selected Options': selected_options})
                st.write(last_df)
        # st.write("Selected Options for Each Element:", selected_options)
    




    # We are having column names and their replacement
    # columns_with_na
    # selected_options

        dum_df = x.copy()
        for index, element in enumerate(columns_with_na):
            if(selected_options[index] == "Mean"):
                dum_df[element].fillna(dum_df[element].mean(), inplace=True)
                # st.write(selected_options[index])
                # st.write("replaced with mean")
            elif(selected_options[index] == "Median"):
                dum_df[element].fillna(dum_df[element].median(), inplace=True)
                # st.write(selected_options[index])
                # st.write("replaced with mode")
            elif(selected_options[index] == "Mode"):
                dum_df[element].fillna(dum_df[element].mode()[0], inplace=True)
                # st.write("replaced with mode")
        st.write(dum_df.head(5))
    
    ##############################


    sub_col = st.columns(7)
    with sub_col[3]:
        st.button("Submit", on_click=navigate, args=("page 5",))
        st.session_state["filtered_df2"] = dum_df

    # We need to handle Categorical data and non categorical data seperately
    # Handle non - categorical data

    
    
    # Handle categorical
        # Text - We need to convert all the letters to upper case
            # 1. Remove the Row
            # 2. Replace with mode






def page5():
    st.button("Go Back", on_click=navigate, args=("page 4",))  # Use on_click
    st.title("Encoding textual columns to numerical",anchor=False)
    mappings = {}
    df = st.session_state['filtered_df2']
    columns_info = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': [get_readable_dtype(df[col].dtype) for col in df.columns],
    })
    text_columns = columns_info.loc[columns_info['Data Type'] == 'Text', 'Column Name'].tolist()
    # Print the result
    # st.write("Columns with text data type:", text_columns)
    
    # csv_file_path = "columns_info.csv"  # Specify your desired file path
    # columns_info.to_csv(csv_file_path, index=False)

    # csv_file_path = "filtered_df2.csv"  # Specify your desired file path
    # df.to_csv(csv_file_path, index=False)

    selected_options = []
    # Display the result
    columns_with_na = text_columns
    he_cols = st.columns([3,1])
    dummy_df = df.copy()
    if(len(columns_with_na) == 0 ):
        st.write("No Textual columns")
    elif(len(columns_with_na) == 1 and columns_with_na[0] == st.session_state["target"]):
        st.info("Peformed label encoding on the target column")        
    else:
            # st.write(columns_with_na)
        elements = columns_with_na
        he_cols[0].subheader("Choose a encoding Measures",anchor=False)

            # Options for the radio buttons
        options = ["Label Encoding", "One Hot Encoding"]
        selected_options = []
        ##-------
        i = 0
        ui_col = st.columns(4)
            # with ui_col[3]:
        he_cols[1].title("Choosed values ",anchor=False)
        while i < len(elements):
            with ui_col[0]:
                if(i==len(elements)):
                    break
                element = elements[i]
                if(element == st.session_state["target"]):
                    selected_options.append("Label Encoding")
                    break
                selected_option = st.radio(
                            f"Select for {element}",
                            options,
                            key=f"radio_{i}",  # Unique key combining the element index
                            index=0  # Default selection (optional)
                        )
                selected_options.append(selected_option)
                    
                    # Show the selected option for the current element
                st.write(f"Selected: {selected_option}")
                st.write("---")
                i += 1
            with ui_col[1]:
                if(i==len(elements)):
                    break
                element = elements[i]
                if(element == st.session_state["target"]):
                    selected_options.append("Label Encoding")
                    break
                selected_option = st.radio(
                            f"Select for {element}",
                            options,
                            key=f"radio_{i}",  # Unique key combining the element index
                            index=0  # Default selection (optional)
                        )
                selected_options.append(selected_option)
                    
                    # Show the selected option for the current element
                st.write(f"Selected: {selected_option}")
                st.write("---")
                i += 1
            with ui_col[2]:
                if(i==len(elements)):
                    break
                element = elements[i]
                
                if(element == st.session_state["target"]):
                    selected_options.append("Label Encoding")
                    break
                selected_option = st.radio(
                            f"Select for {element}",
                            options,
                            key=f"radio_{i}",  # Unique key combining the element index
                            index=0  # Default selection (optional)
                        )
                selected_options.append(selected_option)
                    
                    # Show the selected option for the current element
                st.write(f"Selected: {selected_option}")
                st.write("---")
                i += 1
            # last_df = pd.DataFrame([columns_with_na,selected_options])
        
            

        if(len(text_columns) > 0):
            with ui_col[3]:
                last_df = pd.DataFrame({'Columns with text': columns_with_na,'Selected Options': selected_options})
                st.write(last_df)
    # Apply the selected encoding
    # dummy_df = 
        dummy_df = df.copy()
        # for idx, col in enumerate(columns_with_na):
        #     if selected_options[idx] == "Label Encoding":
        #         dummy_df[col] = dummy_df[col].astype("category").cat.codes
        #     elif selected_options[idx] == "One Hot Encoding":
        #         col_position = dummy_df.columns.get_loc(col)
        #         # Perform one-hot encoding for the current column
        #         one_hot = pd.get_dummies(dummy_df[col], prefix=col)
        #         # Insert the new one-hot encoded columns at the original column position
        #         dummy_df = pd.concat(
        #             [dummy_df.iloc[:, :col_position], one_hot, dummy_df.iloc[:, col_position + 1:]],
        #             axis=1
        #         )
        # st.write("Encoded DataFrame:")
        # st.dataframe(dummy_df)
        
        for idx, col in enumerate(columns_with_na):
            if selected_options[idx] == "Label Encoding":
                # Create mapping for label encoding
                label_mapping = {v: k for k, v in enumerate(dummy_df[col].dropna().unique())}
                mappings[col] = label_mapping
                # Apply label encoding
                dummy_df[col] = dummy_df[col].map(label_mapping)
            elif selected_options[idx] == "One Hot Encoding":
                # Perform one-hot encoding
                col_position = dummy_df.columns.get_loc(col)
                one_hot = pd.get_dummies(dummy_df[col], prefix=col)
                dummy_df = pd.concat(
                    [dummy_df.iloc[:, :col_position], one_hot, dummy_df.iloc[:, col_position + 1:]],
                    axis=1
                )
                mappings[col] = list(one_hot.columns)  # Store one-hot column names

        # # Display Encoded DataFrame
        # st.write("Encoded DataFrame:")
        # st.dataframe(dummy_df)

        # # Display Mappings
        # st.write("Mappings:")
        # for col, mapping in mappings.items():
        #     st.write(f"**{col}**:")
        #     st.write(mapping)

    # st.button("Submit", on_click=navigate, args=("page 6",))
    


    col1, col2, col3,col4  =  st.columns([0.7,3,1, 1])

    st.session_state["encoded_df"] = dummy_df.copy()
    st.session_state["mappings"] = mappings.copy()
    st.write("Encoded DataFrame:")
    st.dataframe(st.session_state["encoded_df"].head(5))
    with col1:
        x =st.button("Show mapping", use_container_width=False) 
        # if(st.button("Show mapping", use_container_width=False)):
        # Display the encoding mapping for each column
    with col2:    
        st.button("Next", on_click=navigate, args=("page 6",))  # Use on_click
    with col3:
        st.write("")
    with col4:
        st.write("")

    
    if(x==True):
        if(len(mappings) == 0):
            st.warning("No columns encoded")
        else:
            for column, mapping in mappings.items():
                st.write("Encoding Mapping for "+column+" : " )
                st.write(mapping)

    st.session_state["mappings"] = mappings.copy()
    # print(mappings)
    


# Another Page
def page6():
    st.button("Go Back", on_click=navigate, args=("page 5",))  # Use on_click

    st.title("Text columns -> Numerical columns",anchor=False)
    # st.header("In this page, we will ask the users to replace the na, null values")
    filtered_df = st.session_state["filtered_df2"]


    col1, col2= st.columns(2)
    
    with col1:
        st.write("Original dataframe")
        st.write(filtered_df.head(5))


    with col2:
        st.write("Numerical dataframe")
        st.write(st.session_state["encoded_df"].head(5))
    encoding_mapping = st.session_state["mappings"].copy()

    # Display the updated DataFrame with encoded columns

    #------ Store encoded and mappings

    col1, col2, col3,col4  =  st.columns([0.7,3,1, 1])

    
    with col1:
        x =st.button("Show mapping", use_container_width=False) 
        # if(st.button("Show mapping", use_container_width=False)):
        # Display the encoding mapping for each column
    with col2:    
        st.button("Next", on_click=navigate, args=("page 7",))  # Use on_click
    with col3:
        st.write("")
    with col4:
        st.write("")

    
    if(x==True):
        if(len(encoding_mapping) == 0):
            st.warning("No columns encoded")
        else:
            for column, mapping in encoding_mapping.items():
                st.write("Encoding Mapping for "+column+" : " )
                st.write(mapping)
    

def page7():
    # In page 7, The user need to select the scaling
    # we have encoded_df and mappings
    st.button("Go Back", on_click=navigate, args=("page 6",))  # Use on_click
    st.title("Select a Scaling Method",anchor=False)

    df = st.session_state["encoded_df"].copy()
    # Display the original DataFrame
    df = df.iloc[:, :-1]

    # Create a radio button to select scaling method
    scaling_methods = ["MinMaxScaler", "StandardScaler", "RobustScaler"]
    selected_scaler = st.radio(" ", scaling_methods, label_visibility="collapsed")
    scaler = None
    # Apply the selected scaling method
    if selected_scaler == "MinMaxScaler":
        scaler = MinMaxScaler()
        st.session_state["scaler"] = scaler
    elif selected_scaler == "StandardScaler":
        scaler = StandardScaler()
        st.session_state["scaler"] = scaler
    elif selected_scaler == "RobustScaler":
        scaler = RobustScaler()
        st.session_state["scaler"] = scaler
    # Perform scaling


    st.session_state["scaler_file"] = scaler
    
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    # Display the scaled DataFrame
    


    col1,col2 = st.columns(2);
    with col1:
        st.write("### Original DataFrame")
        st.dataframe(df.head(5))
    with col2:
        st.write(f"### DataFrame After Applying {selected_scaler}")
        st.dataframe(scaled_df.head(5))

    # Optional: Download the scaled DataFrame
    

    c1, c2, c3,c4  =  st.columns([1.3,3,1, 1])

    
    # Save it using pickle
    # with open("scaler3.pkl", "wb") as f:
    #     pickle.dump(scaler, f)

    st.session_state["scaled_df"] = scaled_df
    with c1:
        csv = scaled_df.to_csv(index=False).encode('utf-8')
        st.download_button(
        label="Download Scaled DataFrame",
        data=csv,
        file_name="scaled_dataframe.csv",
        mime="text/csv"
        )
        # if(st.button("Show mapping", use_container_width=False)):
        # Display the encoding mapping for each column
    with c2:
        st.button("Next", on_click=navigate, args=("page 8",))  # Use on_click
    with c3:
        st.write("")
    with c4:
        st.write("")


def page8():
    # In page 7, The user need to select the scaling
    # we have encoded_df and mappings
    st.button("Go Back", on_click=navigate, args=("page 7",))  # Use on_click
    st.title("Select The Quantity For Training Data ",anchor=False)

    train_perc = st.slider("How Much Training Data ?", 0, 100, 70)
    random_st = st.number_input("Random state",min_value=0,max_value=100,value=42);


    df = st.session_state["encoded_df"]
    df2 = st.session_state["scaled_df"]

    x = df2.iloc[:, :].values
    y = df.iloc[:, -1].values

    try:
        x_train,x_test,y_train,y_test = train_test_split(x,y, train_size= train_perc/100 , random_state= random_st, stratify = y)
    except:
        st.error("Insufficient amount of training data, ' Increase training data to solve the issue' ")
        
        return



    st.button("Next", on_click=navigate, args=("page 9",))  # Use on_click

    st.session_state["x_train"] = x_train
    st.session_state["x_test"] = x_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    # st.write(x_train)
    # st.write(x_test)
    # st.write(y_train)
    # st.write(y_test)

if "high_index" not in st.session_state:
    st.session_state["high_index"] = "None"

def page9():
    # In page 7, The user need to select the scaling
    # we have encoded_df and mappings
    st.button("Go Back", on_click=navigate, args=("page 8",))  # Use on_click
    
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    xgb_classifier = xgb.XGBClassifier()
    X_train = st.session_state["x_train"]
    y_train = st.session_state["y_train"]
    X_test = st.session_state["x_test"]
    y_test = st.session_state["y_test"] 
    
    #-- working code
    # lr.fit(x_train,y_train)
    # y_pred_lr = lr.predict(x_test)
    # # st.write(classification_report(y_test,y_pred_lr))
    # score_lr = round(accuracy_score(y_test,y_pred_lr)*100,2)
    # st.info("Accuracy score using logistic regression "+str(score_lr)+" %")

    # model_file = pickle.dumps(lr)
    # final_ml_model = 'final_ml_model.pkl'
    # pickle.dump(lr,open(final_ml_model,'wb'))

    # # Create a download button for the model file
    # st.download_button(
    #     label="Download Trained Model",         # Button label
    #     data=model_file,                        # The binary content of the model
    #     file_name="final_ml_model.pkl",    # File name for the model
    #     mime="application/octet-stream")
    #-- working code --^
    from sklearn.naive_bayes import GaussianNB
    # Algorithm options
    algorithms = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Support Vector Machine": SVC(kernel = 'linear'),
        "XG Boost Classifier": xgb.XGBClassifier(),
        "Naive Bayes": GaussianNB()
    }


    display_button = False
    # Form to select algorithms
    st.title("Select Algorithm/s",anchor=False)
    selected_algorithms = []
    cols = st.columns(3)
    i = 0
    for algo in algorithms.keys():
        with cols[i%3]:
            if st.checkbox(algo):
                try:
                    selected_algorithms.append(algo)
                    display_button= True
                except:
                    st.error("Invalid data given")
            i = i+1

    st.header("Download File/s ",anchor=False)
    cols = st.columns(3)
    scaler_filename = "scaler.pkl"
    with open(scaler_filename, "wb") as f:
        pickle.dump(st.session_state["scaler_file"], f)

    mapping = st.session_state["mappings"]
    import io
    # Serialize the mapping variable
    mapping_pkl = io.BytesIO()
    pickle.dump(mapping, mapping_pkl)
    mapping_pkl.seek(0)
    # Create a download button
    with cols[0]:
        st.download_button(
        label="Download Mapping Variable",
        data=mapping_pkl,
        file_name="mapping.pkl",
        mime="application/octet-stream"
    )
    # Function to create a download link
    def get_download_link(file_path, file_label):
        with open(file_path, "rb") as file:
            b64 = base64.b64encode(file.read()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{file_path}">{file_label}</a>'
            return href

    if display_button==False:
        return
    with cols[1]:
        st.markdown(get_download_link(scaler_filename, "Download Scaler (.pkl)"), unsafe_allow_html=True)
    # Train models and display results

    results = []
    download_links = {}

    for algo_name in selected_algorithms:
        try:
            model = algorithms[algo_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)    
        except:
            st.error("Only one target, Invalid data ")
            return
        st.session_state["model"].append(model)

        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Save model using pickle
        model_file = BytesIO()
        pickle.dump(model, model_file)
        model_file.seek(0)

        download_links[algo_name] = model_file

        # Store results
        results.append({
            "Algorithm Name": algo_name,
            "Accuracy": round(accuracy, 2),
            "Precision": round(precision, 2),
            "F1 Score": round(f1, 2),
        })

    # Save results in session state
    st.session_state.results_df = pd.DataFrame(results)
    st.session_state.download_links = download_links
    # print(st.session_state["model"])
    with cols[2]:
        st.button("Next", on_click=navigate, args=("page 10",))  # Use on_click
    # import matplotlib.pyplot as plt
    df = pd.DataFrame(results)
    # # Create a bar chart
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.bar(df["Algorithm Name"], df["Accuracy"], color=['blue', 'green', 'red', 'orange'])
    # ax.set_xlabel("Algorithms")
    # ax.set_ylabel("Accuracy")
    # ax.set_title("Algorithm Accuracy Comparison")
    # ax.set_ylim(0, 1)  # Set y-axis limit from 0 to 1
    # plt.xticks(fontsize=10)  # Rotate x-axis labels for better visibility

    # # Display the chart in Streamlit
    # st.pyplot(fig)
    df["Accuracy"] = df["Accuracy"] * 100

    st.title("  ",anchor=False)
    import altair as alt
    chart = (
    alt.Chart(df)
    .mark_bar(cornerRadius=3)
    .encode(
        x=alt.X("Algorithm Name", sort="-y", title="Machine Learning Algorithms"),
        y=alt.Y("Accuracy", scale=alt.Scale(domain=[0, 100]), title="Accuracy (%)"),
        color=alt.Color("Algorithm Name", legend=None),
    )
    .properties( title=alt.TitleParams(
            text="Model Accuracy Comparison",
            anchor="middle",  # Centers the title
            fontSize=26,  # Adjust title font size
        ), width=600, height=600)
)
    
     #Add a full black border using configure
    chart = chart.configure_view(
    stroke="black",  # Black border around the graph
    strokeWidth=2  # Border thickness
).configure_axis(
    domainColor="black",  # X and Y axis in black
    gridColor="gray",  # Light gray grid
    labelColor="black",  # Axis labels in black
    titleColor="black"  # Axis titles in black
).configure_title(
    color="black",  # Title in black
    fontSize=18
)

    # Display chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

    # df.set_index("Algorithm Name", inplace=True)
    # # Display the accuracy bar chart
    # st.bar_chart(df["Accuracy"],  use_container_width=True)

    # If results exist, display table
    if st.session_state.results_df is not None:
        st.write("### Model Performance Table")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.write("**Algorithm Name**")
        col2.write("**Accuracy**")
        col3.write("**Precision**")
        col4.write("**F1 Score**")
        col5.write("**Download**")
        local_high = 0
        for i in range(len(st.session_state.results_df)):
            col1.write(st.session_state.results_df.iloc[i]["Algorithm Name"])
            col2.write(st.session_state.results_df.iloc[i]["Accuracy"])
            col3.write(st.session_state.results_df.iloc[i]["Precision"])
            col4.write(st.session_state.results_df.iloc[i]["F1 Score"])
            col5.download_button(
                label="Download",
                data=st.session_state.download_links[st.session_state.results_df.iloc[i]["Algorithm Name"]],
                file_name=f"{st.session_state.results_df.iloc[i]['Algorithm Name']}.pkl",
                mime="application/octet-stream",
                key=f"download_{i}"  # Unique key to prevent rerun issues
            )
            if(st.session_state.results_df.iloc[i]["Accuracy"] > st.session_state.results_df.iloc[local_high]["Accuracy"]):
                local_high = i
            
        st.session_state["high_index"] = local_high
        st.success("Highest Accuracy : "+st.session_state.results_df.iloc[local_high]["Algorithm Name"])


if "df2" not in st.session_state:
    st.session_state["df2"] = "None"

def page10():
    st.button("Back", on_click=navigate, args=("page 9",))  # Use on_click


    x = st.session_state['final_df_dep_col_all'].columns.tolist()
    x.remove(st.session_state["target"])
    # print(x)

    # st.title("Dynamic Form Generator")

    # Form inputs
    form_data = {}
    # columns_info = st.session_state["columns_info"]
    # mapping = st.session_state["mappings"]

    # print(columns_info)
    # print(mapping)



    df_info = st.session_state["columns_info"].drop('Unique Values', axis=1).copy()
    encoding_map = st.session_state["mappings"]
    df = df_info.copy()
    df = df[df['Column Name'].isin(x)]
    
    # print(df)
    df_info = df.copy()
    # df_info = df_info.drop('Unique Values', axis=1)
    # df_info.drop('Unique Values', axis=1, inplace=True)
    df_info = df_info.to_records(index=False)
    # print(df_info)
    # print(encoding_map)



    # Streamlit Form
    st.title("Dynamic Input Form",anchor=False)
    form_data = {}

    with st.form("user_input_form"):
        for column, dtype in df_info:
            if dtype == "Integer":
                form_data[column] = st.number_input(f"Enter {column}", step=1, format="%d")
            elif dtype == "Float":
                form_data[column] = st.number_input(f"Enter {column}", format="%f")
            elif dtype == "Text" and column in encoding_map:
                options = encoding_map[column] if isinstance(encoding_map[column], list) else list(encoding_map[column].keys())
                form_data[column] = st.selectbox(f"Select {column}", options)

        submitted = st.form_submit_button("Submit")

    if submitted:
        # Convert text selections to encoded values
        processed_data = []
        for column, dtype in df_info:
            if dtype in ["Integer", "Float"]:
                processed_data.append(form_data[column])
            elif dtype == "Text" and column in encoding_map:
                if isinstance(encoding_map[column], dict):
                    processed_data.append(encoding_map[column][form_data[column]])  # Dictionary-based mapping
                elif isinstance(encoding_map[column], list):
                    encoded_value = [1 if form_data[column] == val else 0 for val in encoding_map[column]]
                    processed_data.extend(encoded_value)  # One-hot encoding

        # st.write("### Encoded Data Output:", processed_data)
        # We have processed data, now send this to lr model, and return the output

        model = st.session_state["model"]
        scaler = st.session_state["scaler"]
        input_data = tuple(processed_data)

        #chaning the input data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)
        #reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        std_data = scaler.transform(input_data_reshaped)
        # -- std_data = scaler.transform(input_data_reshaped)
        # -- pred = loaded_model.predict(std_data)
        x = st.session_state["model"]
        try:
            model = x[st.session_state["high_index"]]
            # print (st.session_state["model"])
        except:
            st.error("model is not trained")
            return
        try:
            pred = model.predict(std_data)
            
        except:
            st.error("Invalid data found, Start from HOME :)")
            st.button("Go Home", on_click=navigate, args=("Home",))  # Use on_click
            return
        # st.success(pred)
        target = st.session_state["target"]
        if(target in encoding_map):
            def decode_value(mapping, category, encoded_value):
                reverse_map = {v: k for k, v in mapping[category].items()}  # Reverse dictionary
                return reverse_map.get(encoded_value, "Unknown")  # Get actual value

            # Example usage
            encoded_value = pred[0]  # Example encoded value

            
            category = st.session_state["target"]  # Example category
            decoded_value = decode_value(encoding_map, category, encoded_value)
            st.success(decoded_value)
        else:
            st.success(pred[0])
        


def page1():
    # Function to Encode Local Image to Base64
    def get_base64_of_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    image_base64 = get_base64_of_image("rrt8.jpg")
    # Custom CSS for Gradient Title and Button



    gradient_css = """
        <style>
            
            .gradient-text {

                font-size: 40px;
                font-weight: bold;
                text-align:center;


            color: #11193e; 
                background-image: linear-gradient(45deg, #11193e 58%, #2a347a 55%); 
                background-clip: text; 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            width:100%
    }

            /* Centering Content */
            .container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                width: 150%;
                max-width: 900px; /* Adjust width for better balance */
                margin: auto;
                text-align: center;
                padding-top: 50px;
            }

            .subtext {
                font-size: 18px;
                font-weight: 500;
                
                color: #444;
            }

            .stButton > button {background-image: linear-gradient(to right, #16222A 0%, #3A6073  51%, #16222A  100%)}
            .stButton > button{
                margin: 10px;
                padding: 15px 45px;
                text-align: center;
                text-transform: uppercase;
                transition: 0.5s;
                background-size: 200% auto;
                color: white;            
                box-shadow: 0 0 20px #eee;
                border-radius: 10px;
                display: block;
                font-weight: 800;
                font-size: 30px;
            }

            .stButton > button:hover {
                background-position: right center; /* change the direction of the change here */
                color: #fff;
                text-decoration: none;
                
            }

            
            
        </style>
    """

    # Apply the gradient text & button styles
    st.markdown(gradient_css, unsafe_allow_html=True)

    # Centered Content
    st.markdown('<div class="container">', unsafe_allow_html=True)

    # Gradient Title
    st.markdown('<p class="gradient-text">Developing Dynamic Model for Classification and Prediction using Machine Learning</p>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    # Student and Guide Name
    st.markdown('<p class="subtext">Student Name: <b>Manindra Khandyana (2023000473)</b></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtext">Guide Name: <b>Dr. Muktevi Srivenkatesh</b></p>', unsafe_allow_html=True)

    # Spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # "Get Started" Button
    # if st.button("Get Started"):
    #     st.switch_page("main.py")  # Change this to your main app script
    
    st.button("Get Started", on_click=navigate, args=("page 2",))

    pdf_path = "manindra_abstract_2.pdf"  # Ensure this file is in the same directory


    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8") 
    st.markdown("""
        <style>
            .download-btn {
                margin:10px;
                margin-top:0px;
                display: inline-block;
                background: linear-gradient(90deg, #ff7e5f, #feb47b);
                color: white;
                padding: 15px ;
                font-size: 16px;
                font-weight: 800;
                text-align: center;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .download-btn:hover {
                background: linear-gradient(90deg, #feb47b, #ff7e5f);
            }
        </style>
    """, unsafe_allow_html=True)

    # Display the styled button
    pdf_download_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="Manindra Abstract.pdf"><button class="download-btn">📄 Download Abstract</button></a>'
    st.markdown(pdf_download_link, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)



    def set_background(image_file):
        with open(image_file, "rb") as img:
            encoded_string = base64.b64encode(img.read()).decode()

        background_css = f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
        </style>
        """
        st.markdown(background_css, unsafe_allow_html=True)

    # Call the function with the correct image filename
    set_background("rrt8.jpg")  # Ensure this file is in the same directory




if st.session_state["current_page"] == "Home":
    page1()
elif st.session_state["current_page"] == "page 2":
    page2()
elif st.session_state["current_page"] == "page 3":
    page3()
elif st.session_state["current_page"] == "page 4":
    page4()
elif st.session_state["current_page"] == "page 5":
    page5()
elif st.session_state["current_page"] == "page 6":
    page6()
elif st.session_state["current_page"] == "page 7":
    page7()
elif st.session_state["current_page"] == "page 8":
    page8()
elif st.session_state["current_page"] == "page 9":
    page9()
elif st.session_state["current_page"] == "page 10":
    page10()
