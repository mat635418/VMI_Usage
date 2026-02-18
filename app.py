import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="VMI Baseline 2025 & Way Forward", layout="wide")
FILENAME = "VMI_Usage.XLSX"

# --- DATA LOADING FUNCTION ---
@st.cache_data
def load_data():
    try:
        # Tries to load the Excel file. 
        # Ensure 'VMI_Usage.XLSX' is in the root folder.
        df = pd.read_excel(FILENAME)
        return df
    except FileNotFoundError:
        st.error(f"File '{FILENAME}' not found. Please place it in the root directory.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# --- MAIN APP ---
def main():
    st.title("📊 VMI Usage: 2025 Baseline & Way Forward")
    st.markdown("Automated insights and scenario simulation based on 2025 Purchasing Data.")

    df = load_data()
    if df is None:
        return

    # --- PREPROCESSING ---
    # Standardize columns based on your file structure
    # Expected: 'check' (VMI status), 'category', 'PO Quantity' (Volume), 'Supplier name'
    
    # Clean up column names just in case
    df.columns = [c.strip() for c in df.columns]
    
    # Filter for valid units if necessary (Assuming PO Quantity is the KG volume as per prompt context)
    # We will treat 'PO Quantity' as the 'Volume in KG' for analysis
    
    # 1. GLOBAL KPIS
    st.header("1. Current Baseline (2025)")
    
    total_vol = df['PO Quantity'].sum()
    total_pos = df.shape[0]
    
    vmi_df = df[df['check'] == 'VMI']
    vmi_vol = vmi_df['PO Quantity'].sum()
    vmi_pos = vmi_df.shape[0]
    
    vmi_vol_pct = (vmi_vol / total_vol) * 100 if total_vol else 0
    vmi_po_pct = (vmi_pos / total_pos) * 100 if total_pos else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Volume (KG)", f"{total_vol:,.0f}")
    kpi2.metric("Total POs", f"{total_pos:,}")
    kpi3.metric("VMI Adoption (Vol)", f"{vmi_vol_pct:.1f}%")
    kpi4.metric("VMI Adoption (POs)", f"{vmi_po_pct:.1f}%")

    # --- CHARTS ROW 1 ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Volume Split (KG)")
        vol_split = df.groupby('check')['PO Quantity'].sum().reset_index()
        fig_vol = px.pie(vol_split, values='PO Quantity', names='check', 
                         color='check', color_discrete_map={'VMI':'#00CC96', 'not VMI':'#EF553B'})
        st.plotly_chart(fig_vol, use_container_width=True)

    with col2:
        st.subheader("PO Count Split")
        po_split = df['check'].value_counts().reset_index()
        po_split.columns = ['check', 'count']
        fig_po = px.pie(po_split, values='count', names='check', 
                        color='check', color_discrete_map={'VMI':'#00CC96', 'not VMI':'#EF553B'})
        st.plotly_chart(fig_po, use_container_width=True)

    # --- CATEGORY ANALYSIS ---
    st.divider()
    st.header("2. Category Deep Dive")
    
    # Aggregate data by Category
    cat_agg = df.groupby('category').agg(
        Total_Vol=('PO Quantity', 'sum'),
        VMI_Vol=('PO Quantity', lambda x: x[df.loc[x.index, 'check'] == 'VMI'].sum())
    ).reset_index()
    cat_agg['VMI %'] = (cat_agg['VMI_Vol'] / cat_agg['Total_Vol']) * 100
    cat_agg = cat_agg.sort_values('VMI %', ascending=True)

    col3, col4 = st.columns([1, 2])
    
    with col3:
        st.dataframe(cat_agg.style.format({'Total_Vol': '{:,.0f}', 'VMI_Vol': '{:,.0f}', 'VMI %': '{:.1f}%'}), use_container_width=True)

    with col4:
        fig_bar = px.bar(cat_agg, x='category', y=['VMI_Vol', 'Total_Vol'], 
                         barmode='overlay', title="VMI Volume vs Total Volume by Category")
        fig_bar.update_traces(opacity=0.7)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- SIMULATION: WAY FORWARD ---
    st.divider()
    st.header("3. Way Forward Simulation (Option 7)")
    st.markdown("Simulate moving the **Top X** non-VMI suppliers in specific categories to VMI to see the impact on adoption rates.")

    # Dropdown to select category
    selected_category = st.selectbox("Select Category to Simulate:", df['category'].unique())
    
    # Filter data for selected category
    cat_df = df[df['category'] == selected_category]
    
    # Group by Supplier to find Top Non-VMI opportunities
    supp_agg = cat_df.groupby('Supplier name').agg(
        Total_Vol=('PO Quantity', 'sum'),
        VMI_Vol=('PO Quantity', lambda x: x[cat_df.loc[x.index, 'check'] == 'VMI'].sum())
    ).reset_index()
    
    supp_agg['Non_VMI_Vol'] = supp_agg['Total_Vol'] - supp_agg['VMI_Vol']
    supp_agg['Current_VMI_Status'] = supp_agg.apply(lambda r: "100% VMI" if r['Non_VMI_Vol'] == 0 else ("Partial" if r['VMI_Vol'] > 0 else "Non-VMI"), axis=1)
    
    # Sort by Total Volume (Influence)
    supp_agg = supp_agg.sort_values('Total_Vol', ascending=False).reset_index(drop=True)

    # Simulation Slider
    st.subheader(f"Simulate: {selected_category}")
    top_x = st.slider(f"Move Top X Suppliers to 100% VMI (Impact Analysis)", 1, 10, 5)

    # --- LOGIC ---
    # Identify the Top X suppliers (by volume) regardless of current status, 
    # because standardizing Top X implies ensuring they are ALL 100% VMI.
    target_suppliers = supp_agg.head(top_x)
    
    # Calculate potential gain
    # Gain = The Non-VMI volume of these Top X suppliers
    potential_gain_vol = target_suppliers['Non_VMI_Vol'].sum()
    
    current_cat_total = cat_agg[cat_agg['category'] == selected_category]['Total_Vol'].values[0]
    current_cat_vmi = cat_agg[cat_agg['category'] == selected_category]['VMI_Vol'].values[0]
    
    new_vmi_vol = current_cat_vmi + potential_gain_vol
    new_vmi_pct = (new_vmi_vol / current_cat_total) * 100

    # Display Simulation Results
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        st.metric(f"Current VMI % ({selected_category})", f"{(current_cat_vmi/current_cat_total)*100:.1f}%")
        st.metric(f"Projected VMI % (Top {top_x} Suppliers)", f"{new_vmi_pct:.1f}%", delta=f"{new_vmi_pct - (current_cat_vmi/current_cat_total)*100:.1f}%")
        
        st.markdown(f"**Insight:** Negotiating with the TOP {top_x} suppliers will push {selected_category} adoption from **{(current_cat_vmi/current_cat_total)*100:.0f}%** to **{new_vmi_pct:.0f}%**.")

    with col_sim2:
        st.write(f"**Target Suppliers (Top {top_x}):**")
        st.dataframe(target_suppliers[['Supplier name', 'Total_Vol', 'Current_VMI_Status']].style.format({'Total_Vol': '{:,.0f}'}))

    # --- GLOBAL SIMULATION CHART ---
    # Show "What if we did this for ALL categories?"
    st.subheader("Global Impact: Apply Top 5 Strategy Across All Categories")
    
    global_sim_data = []
    
    for cat in df['category'].unique():
        c_df = df[df['category'] == cat]
        s_agg = c_df.groupby('Supplier name')['PO Quantity'].sum().reset_index().sort_values('PO Quantity', ascending=False)
        
        # Get Top 5 suppliers for this category
        top5_suppliers = s_agg.head(5)['Supplier name'].tolist()
        
        # Calculate totals
        cat_total = c_df['PO Quantity'].sum()
        cat_current_vmi = c_df[c_df['check'] == 'VMI']['PO Quantity'].sum()
        
        # Calculate simulated VMI (Assuming Top 5 become 100% VMI)
        # Any volume from Top 5 counts as VMI
        vol_from_top5 = c_df[c_df['Supplier name'].isin(top5_suppliers)]['PO Quantity'].sum()
        # Any volume from Others that is ALREADY VMI counts as VMI
        vol_from_others_vmi = c_df[(~c_df['Supplier name'].isin(top5_suppliers)) & (c_df['check'] == 'VMI')]['PO Quantity'].sum()
        
        simulated_vmi = vol_from_top5 + vol_from_others_vmi
        
        global_sim_data.append({
            'Category': cat,
            'Current VMI %': (cat_current_vmi / cat_total) * 100,
            'Projected VMI %': (simulated_vmi / cat_total) * 100
        })
    
    sim_df = pd.DataFrame(global_sim_data)
    
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Bar(x=sim_df['Category'], y=sim_df['Current VMI %'], name='Current'))
    fig_sim.add_trace(go.Bar(x=sim_df['Category'], y=sim_df['Projected VMI %'], name='With Top 5 Strategy'))
    fig_sim.update_layout(title="Impact of Moving Top 5 Suppliers to 100% VMI per Category", barmode='group')
    st.plotly_chart(fig_sim, use_container_width=True)

if __name__ == "__main__":
    main()
