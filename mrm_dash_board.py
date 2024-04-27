# from cProfile import label
import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_dynamic_filters import DynamicFilters
import altair as alt
import cufflinks as cf

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

@st.cache_data
def load_csv_data(file_nm):
    data = pd.read_csv(file_nm)
    return data

@st.cache_data
def get_bar_chart(dataset, x_var_nm, y_var_nm):
    fig = px.bar(dataset, x=x_var_nm, y=y_var_nm, height=400)
    return fig

def groupfct(dataset, vr_nm):
    col_list = ['Name of Model']
    col_list.append(vr_nm)
    dat = dataset[col_list]
    dat.drop_duplicates(col_list, inplace = True)
    dat_agg = dat.groupby(vr_nm).aggregate(cnt = pd.NamedAgg(column='Name of Model', aggfunc='count')).reset_index()
    return dat_agg

data_path = "https://github.com/poudas1981/Model_Risk_Management_Inventory/tree/main/MRM_FAKE_DATA.csv?raw=true"


### Load dataset ###
# data = load_csv_data(file_nm = data_path)
data = pd.read_csv(data_path)
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
drop_var = ['Key Field',	'Model ID Number',	'Model Sub-ID',	'Name of Model',	'Submodel',	'Feeder Model',	'Data Source', 'Comments', 'Inventory Date', 'Implementation Date']
collist = data.drop(drop_var, axis=1).columns.tolist()
dat3 = data[['Model Owner', 'Risk Rank', 'Model ID Number']]
table = pd.pivot_table(dat3, values= 'Model ID Number', index=['Model Owner'],
                       columns=['Risk Rank'], aggfunc="count").reset_index()
dat3_agg = dat3.groupby(['Model Owner', 'Risk Rank']).aggregate(cnt = pd.NamedAgg(column = 'Model ID Number', aggfunc = 'count')).reset_index()
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

with graph:
    st.title('Summary Statistics')
    tab1, tab2, tab3 = st.tabs(["Data Analysis", "Data Filtering", "Model Owner"])
    with tab1:
        col11, col12 = st.columns(2)
        bar_axis = st.sidebar.selectbox(label="Bar Chart Model Type", options=collist, placeholder ='Risk Rank')
        if bar_axis:
            st.sidebar.title(f'Bar Chart: {bar_axis}')
            agg_data = groupfct(dataset = data, vr_nm = bar_axis)
            bar_fig = get_bar_chart(dataset = agg_data, x_var_nm = bar_axis, y_var_nm = 'cnt')
            pie_fig = agg_data.iplot(kind="pie", labels=bar_axis, values="cnt",
                         title=f"Distribution Per {bar_axis}",
                         hole=0.4,
                         asFigure=True)
        else:
            st.sidebar.title('Bar Chart: Risk Rank')
            agg_data = groupfct(dataset = data, vr_nm = 'Risk Rank')
            bar_fig = get_bar_chart(dataset = agg_data, x_var_nm = 'Risk Rank', y_var_nm = 'cnt')
            pie_fig = agg_data.iplot(kind="pie", labels='Risk Rank', values="cnt",
                         title=f"Distribution Per Risk Rank",
                         hole=0.4,
                         asFigure=True)
        x = st.slider(label ='Select the Number of rows ro display', min_value=5, max_value=30)
        st.write(data.head(x))
        with col11:
            bar_fig
        with col12:
            pie_fig
    with tab3:
        col11, col12 = st.columns(2)
        with col11:
            st.altair_chart(chart,use_container_width=True,theme="streamlit")
    with tab2:
        dynamic_filters2 = DynamicFilters(data, filters=collist)
        dynamic_filters2.display_filters(location='sidebar')
        dynamic_filters2.display_df()
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
