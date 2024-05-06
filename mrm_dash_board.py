# from cProfile import label
import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_dynamic_filters import DynamicFilters
import altair as alt
import cufflinks as cf
import requests
import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from datetime import date

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

def filter_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    st.markdown("### Filter based on validation date.")
    modify2 = st.checkbox("Select Variable")

    if not modify2:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    datecol = []
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
            datecol.append(col)

    modification_container = st.container()
    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", datecol)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))  
            user_date_input = right.date_input(
                f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
            if len(user_date_input) == 2:
                user_date_input = tuple(map(pd.to_datetime, user_date_input))
                start_date, end_date = user_date_input
                df = df.loc[df[column].between(start_date, end_date)] 
    return df 

def model_owner(df):
    nm_list = df['Model Owner'].unique()
    name_ = st.selectbox(label='Select your Name', options=nm_list)
    dat = df[df['Model Owner'] == name_]
    dat = dat[['Name of Model', 'Risk Rank', 'Next Validation Start Date']]
    dat.drop_duplicates(['Name of Model', 'Risk Rank', 'Next Validation Start Date'], inplace=True)
    dat['Next Validation Start Date'] = pd.to_datetime(dat['Next Validation Start Date'])
    dat['today'] = date.today()
    dat['today'] = pd.to_datetime(dat['today'])
    diff = []
    for i in range(dat.shape[0]):
        delta = dat['Next Validation Start Date'].tolist()[i] - dat['today'].tolist()[i]
        diff.append(delta.days)
    dat['day_diff'] = diff
    dat.sort_values(by = 'day_diff', ascending = False, inplace=True)
    return st.write(dat)
def data_plot(df):
    collist = df.columns.tolist()
    bar_axis = st.selectbox(label="Bar Chart Model Type", options=collist, placeholder ='Risk Rank')
    if bar_axis:
        st.title(f'Bar Chart: {bar_axis}')
        agg_data = groupfct(dataset = df, vr_nm = bar_axis)
        bar_fig = get_bar_chart(dataset = agg_data, x_var_nm = bar_axis, y_var_nm = 'cnt')
        pie_fig = agg_data.iplot(kind="pie", labels=bar_axis, values="cnt",
                        title=f"Distribution Per {bar_axis}",
                        hole=0.4,
                        asFigure=True)
    else:
        st.title('Bar Chart: Risk Rank')
        agg_data = groupfct(dataset = df, vr_nm = 'Risk Rank')
        bar_fig = get_bar_chart(dataset = agg_data, x_var_nm = 'Risk Rank', y_var_nm = 'cnt')
        pie_fig = agg_data.iplot(kind="pie", labels='Risk Rank', values="cnt",
                        title=f"Distribution Per Risk Rank",
                        hole=0.4,
                        asFigure=True)
    modification_container = st.container()
    with modification_container:
        col11, col22 = st.columns(2)
        with col11:
            bar_fig
        with col22:
            pie_fig


@st.cache_data
def load_csv_data(file_nm):
    data = pd.read_csv(file_nm)
    return data

def filter_dataframe(df: pd.DataFrame, Widget_nm) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox(Widget_nm)
    # modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    datecol = []
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
            datecol.append(col)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif column in datecol:
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


@st.cache_data
def get_bar_chart(dataset, x_var_nm, y_var_nm):
    fig = px.bar(dataset, x=x_var_nm, y=y_var_nm, height=400)
    return fig

def groupfct(dataset, vr_nm):
    col_list = ['Name of Model']
    if vr_nm == 'Name of Model':
        dat_agg = pd.DataFrame(data[['Name of Model']].value_counts()).reset_index()
        dat_agg.rename(columns={'count':'cnt'}, inplace=True)
    else:
        col_list.append(vr_nm)
        dat = dataset[col_list]
        dat.drop_duplicates(col_list, inplace = True)
        dat_agg = dat.groupby(vr_nm).aggregate(cnt = pd.NamedAgg(column='Name of Model', aggfunc='nunique')).reset_index()
    return dat_agg

data_path = "C://Users//merli//OneDrive//Streamlit//dataset//MRM_FAKE_DATA.csv"
issue_path = "C://Users//merli//OneDrive//Streamlit//dataset//issue.csv"

### Load dataset ###
data = load_csv_data(file_nm = 'MRM_FAKE_DATA.csv')
issue = pd.read_csv('issue.csv' , encoding='cp1252')
issue.drop(['Issue Number',	'AuditBoard Number'], axis=1, inplace=True)
risk_rating = groupfct(dataset = data, vr_nm = 'Risk Rank')
# st.write(risk_rating)
ML_DATA = groupfct(dataset = data, vr_nm = 'Machine Learning')


st.markdown("## Model Risk Management: :red[Model Inventory] :book: :chart: :bar_chart:")   ## Main Title
st.markdown('#### Facilitate the process of Navigating the model Inventory data. ')
Stat = st.container()
graph = st.container()
tab1, tab2 = st.columns(2)
# st.write(data.head())
########## Bar Chart Logic ##################
drop_var = ['Key Field',	'Model ID Number',	'Model Sub ID',	'Name of Model',	'Submodel',	'Feeder Model',	'Data Source', 'Comments', 'Inventory Date', 'Implementation Date']
collist = data.drop(drop_var, axis=1).columns.tolist()
dat3 = data[['Model Owner', 'Risk Rank', 'Model ID Number']]
table = pd.pivot_table(dat3, values= 'Model ID Number', index=['Model Owner'],
                       columns=['Risk Rank'], aggfunc="count").reset_index()
dat3_agg = dat3.groupby(['Model Owner', 'Risk Rank']).aggregate(cnt = pd.NamedAgg(column = 'Model ID Number', aggfunc = 'nunique')).reset_index()
barfig = table.plot.bar(x='Model Owner', stacked=True, title='The number of model')
chart = alt.Chart(dat3_agg).mark_bar().encode(
        x = alt.X('Model Owner',title="Model Owner",type="nominal")
       ,y=alt.Y("cnt",title="Number of Model")
        ,color= alt.Color("Risk Rank",title="Model owner Distribution by Risk Rank")
    )
 

bar_fig2 = table.iplot(kind="bar",
                        barmode="stack",
                        xTitle="Model Owner",
                        title="Distribution of Risk rating by Model Owner",
                        asFigure=True,
                        opacity=1.0,
                        )
data2 = data.copy()
drop_var2 = ['Key Field', 'Model Sub ID',	'Submodel',	'Feeder Model',	'Data Source', 'Comments', 'Inventory Date', 'Implementation Date']
data2.drop_duplicates(drop_var2, inplace=True)
with graph:
    st.title('Summary Statistics')
    tab1, tab2, tab3,tab4 = st.tabs(["Data Analysis", "Data Filtering", "Model Owner", "Validation Issues"])
    with tab1:
        data_plot(data2)
    with tab2:
        st.dataframe(filter_dataframe(data2, "Add filters"))
    with tab3:
        st.title("Model Owner Play Book")
        st.altair_chart(chart,use_container_width=True,theme="streamlit")
        model_owner(df= data)
        st.dataframe(filter_date(data2))
    with tab4:
        st.markdown("### Model Validation Issues")
        st.dataframe(filter_dataframe(issue, "Select Attributes"))

    # with tab2:
    #     st.dataframe(filter_dataframe(data))
        # dynamic_filters2 = DynamicFilters(data, filters=collist)
        # dynamic_filters2.display_filters(location='sidebar')
        # dynamic_filters2.display_df()
        # dynamic_filters.display_filters()
        # dynamic_filters.display_df()
        # with st.sidebar:
        #     dynamic_filters.display_filters()
        # dynamic_filters.display_df()

    # with tab3:
    #     col111, col122 = st.columns(2)
    #     with col111:
    #         st.markdown('#### Distribution of Model Owner by Risk Rating')
    #         bar_fig2
    #         barfig.set_ylim(0,500)
            
st.button("Rerun")