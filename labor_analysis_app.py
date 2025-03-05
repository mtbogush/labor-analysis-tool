import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import math
from datetime import datetime, time, timedelta

# Set page config
st.set_page_config(
    page_title="Labor Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Application title and description
st.title("Labor Analysis Tool")
st.markdown("Upload an Excel or CSV file with your timesheet data to generate a labor analysis report.")

# Define the functions from your notebook
def get_sorted_hour_columns():
    """Generate sorted hour columns in proper format"""
    hours = []
    current_time = datetime.strptime('5:00 AM', '%I:%M %p')
    
    for _ in range(20):  # 5 AM to 12 AM
        hours.append(current_time.strftime('%I:00 %p'))
        current_time += timedelta(hours=1)
    
    return hours

def calculate_hourly_values(row):
    """Calculate hourly values for a single row based on clock in/out times, accounting for overtime"""
    try:
        date = pd.to_datetime(row['Date']).date()
        
        # Strip whitespace to handle cases like "4:03PM "
        clock_in_time = row['Clock_In_Time'].strip() if isinstance(row['Clock_In_Time'], str) else row['Clock_In_Time']
        clock_out_time = row['Clock_Out_Time'].strip() if isinstance(row['Clock_Out_Time'], str) else row['Clock_Out_Time']
        
        try:
            # Try standard format first
            clock_in = pd.to_datetime(f"{date} {clock_in_time}", format=f"%Y-%m-%d %I:%M %p")
            clock_out = pd.to_datetime(f"{date} {clock_out_time}", format=f"%Y-%m-%d %I:%M %p")
        except ValueError:
            # If that fails, try without space between time and AM/PM
            try:
                clock_in = pd.to_datetime(f"{date} {clock_in_time}", format=f"%Y-%m-%d %I:%M%p")
                clock_out = pd.to_datetime(f"{date} {clock_out_time}", format=f"%Y-%m-%d %I:%M%p")
            except ValueError:
                # Fall back to the most flexible option
                clock_in = pd.to_datetime(f"{date} {clock_in_time}")
                clock_out = pd.to_datetime(f"{date} {clock_out_time}")
        
        # Handle overnight shifts
        if clock_out < clock_in:
            clock_out += timedelta(days=1)
        
        hours = {}
        wage = float(row['Wage'])
        
        # Get regular and overtime hours from the row
        regular_hours = float(row['Regular hours'])
        ot_hours = float(row.get('OT hours', 0)) if 'OT hours' in row else 0
        double_ot_hours = float(row.get('Double OT hours', 0)) if 'Double OT hours' in row else 0
        
        # Calculate transition time from regular to overtime
        # This is when overtime starts (clock_in time + regular hours)
        regular_to_ot_transition = clock_in + timedelta(hours=regular_hours)
        
        # Calculate transition time from overtime to double overtime (if applicable)
        ot_to_double_transition = None
        if double_ot_hours > 0:
            ot_to_double_transition = regular_to_ot_transition + timedelta(hours=ot_hours)
        
        # Start from the beginning of the clock-in hour
        current_time = clock_in.replace(minute=0, second=0, microsecond=0)
        
        while current_time < clock_out:
            hour_key = current_time.strftime('%I:00 %p')
            hour_end = current_time + timedelta(hours=1)
            
            # Calculate the effective work time in this hour
            hour_start_time = max(current_time, clock_in)
            hour_end_time = min(hour_end, clock_out)
            
            # Initialize pay components for this hour
            regular_pay_this_hour = 0
            ot_pay_this_hour = 0
            double_ot_pay_this_hour = 0
            
            # Calculate regular time portion (if any)
            if hour_start_time < regular_to_ot_transition:
                regular_end = min(hour_end_time, regular_to_ot_transition)
                regular_time = (regular_end - hour_start_time).total_seconds() / 3600
                regular_pay_this_hour = regular_time * wage
            
            # Calculate overtime portion (if any)
            if (ot_to_double_transition is None or hour_start_time < ot_to_double_transition) and hour_end_time > regular_to_ot_transition:
                ot_start = max(hour_start_time, regular_to_ot_transition)
                ot_end = hour_end_time if ot_to_double_transition is None else min(hour_end_time, ot_to_double_transition)
                if ot_end > ot_start:  # Ensure positive time
                    ot_time = (ot_end - ot_start).total_seconds() / 3600
                    ot_pay_this_hour = ot_time * wage * 1.5
            
            # Calculate double overtime portion (if any)
            if ot_to_double_transition is not None and hour_end_time > ot_to_double_transition:
                double_ot_start = max(hour_start_time, ot_to_double_transition)
                double_ot_time = (hour_end_time - double_ot_start).total_seconds() / 3600
                double_ot_pay_this_hour = double_ot_time * wage * 2
            
            # Total pay for this hour
            hours[hour_key] = regular_pay_this_hour + ot_pay_this_hour + double_ot_pay_this_hour
            
            # Move to next hour
            current_time = hour_end
        
        return hours
    except Exception as e:
        st.error(f"Error processing row: {row['Employee']} - {e}")
        return {}

def process_timesheet_data(df):
    """Process the timesheet data to calculate hourly values"""
    processed_df = df.copy()
    
    # Get sorted hour columns
    hour_cols = get_sorted_hour_columns()
    
    # Initialize hour columns with zeros
    for col in hour_cols:
        processed_df[col] = 0.0
    
    # Calculate hourly values for each row
    with st.spinner("Processing timesheet data..."):
        progress_bar = st.progress(0)
        total_rows = len(processed_df)
        
        for idx, row in processed_df.iterrows():
            try:
                hourly_values = calculate_hourly_values(row)
                for hour, value in hourly_values.items():
                    if hour in processed_df.columns:
                        processed_df.at[idx, hour] = value
            except Exception as e:
                st.error(f"Error processing row {idx}: {e}")
            
            # Update progress
            progress_bar.progress((idx + 1) / total_rows)
    
    # Ensure columns are in correct order
    other_cols = [col for col in processed_df.columns if col not in hour_cols and col != 'Total pay']
    processed_df = processed_df[other_cols + hour_cols + ['Total pay']]
    
    return processed_df

def create_dashboard(df, department=None, exclude_employee=None):
    """Create a dashboard view for specific department and employee filters"""
    filtered_df = df.copy()
    if department:
        filtered_df = filtered_df[filtered_df['Department'] == department]
    if exclude_employee:
        exclude_name = f"{exclude_employee[1]}, {exclude_employee[0]}"
        filtered_df = filtered_df[filtered_df['Employee'] != exclude_name]
    
    # Get hour columns in correct order
    hour_cols = get_sorted_hour_columns()
    hour_cols = [col for col in hour_cols if col in filtered_df.columns]
    
    # Convert string dollar amounts to numeric
    numeric_cols = hour_cols + ['Total pay']
    for col in numeric_cols:
        if col in filtered_df.columns and filtered_df[col].dtype == 'object':
            filtered_df[col] = pd.to_numeric(filtered_df[col].str.replace('$', '').str.replace(',', ''), errors='coerce')
    
    # Create pivot table
    pivot_df = pd.pivot_table(
        filtered_df,
        values=numeric_cols,
        index='DoW',
        aggfunc='sum'
    )
    
    # Reorder days
    days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    pivot_df = pivot_df.reindex(days_order)
    
    # Ensure columns are in correct order
    pivot_df = pivot_df[hour_cols + ['Total pay']]
    
    return pivot_df

def create_summary(df, department=None, exclude_employee=None):
    """Create a summary view with costs broken down by meal period"""
    filtered_df = df.copy()
    if department:
        filtered_df = filtered_df[filtered_df['Department'] == department]
    if exclude_employee:
        exclude_name = f"{exclude_employee[1]}, {exclude_employee[0]}"
        filtered_df = filtered_df[filtered_df['Employee'] != exclude_name]
    
    def categorize_shift(row):
        """Categorize each time slot into meal periods"""
        dow = row['DoW']
        time_str = row['Clock_In_Time'].strip()  # Add .strip() to remove whitespace
        
        try:
            hour = pd.to_datetime(time_str, format='%I:%M %p').hour
        except ValueError:
            # Try alternative formats if standard format fails
            try:
                # Try without space between time and AM/PM
                hour = pd.to_datetime(time_str, format='%I:%M%p').hour
            except ValueError:
                # Fallback to a more flexible approach
                hour = pd.to_datetime(time_str, errors='coerce').hour
                if pd.isna(hour):
                    # Default to Shareables if we can't parse the time
                    return 'Shareables'
        
        if dow in ['Saturday', 'Sunday']:
            if 5 <= hour < 15:  # 5AM to 2:59PM
                return 'Brunch'
            elif 15 <= hour < 24:  # 3PM to 11:59PM
                return 'Shareables'
        else:  # Weekdays
            if 5 <= hour < 11:  # 5AM to 10:59AM
                return 'Breakfast'
            elif 11 <= hour < 15:  # 11AM to 2:59PM
                return 'Lunch'
            elif 15 <= hour < 24:  # 3PM to 11:59PM
                return 'Shareables'
        
        return 'Shareables'
    
    # Add meal period category
    filtered_df['Meal_Period'] = filtered_df.apply(categorize_shift, axis=1)
    
    # Convert Total pay to numeric if it's not already
    if filtered_df['Total pay'].dtype == 'object':
        filtered_df['Total pay'] = pd.to_numeric(filtered_df['Total pay'].str.replace('$', '').str.replace(',', ''), errors='coerce')
    
    # Create summary by meal period
    summary = pd.pivot_table(
        filtered_df,
        values='Total pay',
        index='Meal_Period',
        aggfunc='sum'
    )
    
    # Add total row
    summary.loc['Total'] = summary.sum()
    
    return summary

def generate_excel_report(df, output_file=None):
    """Generate a single Excel file with multiple sheets for all reports"""
    # Process the data first
    processed_df = process_timesheet_data(df)
    
    reports = {
        'Raw_Data': {'type': 'raw'},
        'Dashboard_FoH': {'dept': 'FOH', 'exclude': None, 'type': 'dashboard'},
        'Dashboard_BoH': {'dept': 'Kitchen', 'exclude': None, 'type': 'dashboard'},
        'Dashboard_BoH_NO_Josh': {'dept': 'Kitchen', 'exclude': ('Josh', 'Ashley'), 'type': 'dashboard'},
        'Summary_FoH': {'dept': 'FOH', 'exclude': None, 'type': 'summary'},
        'Summary_BoH': {'dept': 'Kitchen', 'exclude': None, 'type': 'summary'},
        'Summary_BoH_No_Josh': {'dept': 'Kitchen', 'exclude': ('Josh', 'Ashley'), 'type': 'summary'}
    }
    
    output_buffer = io.BytesIO()
    
    # Configure Excel writer to handle NaN/INF values
    writer_options = {
        'engine': 'xlsxwriter',
        'engine_kwargs': {'options': {'nan_inf_to_errors': True}}
    }
    
    with pd.ExcelWriter(output_buffer if output_file is None else output_file, **writer_options) as writer:
        workbook = writer.book
        
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'bg_color': '#D9D9D9',
            'border': 1
        })
        
        currency_format = workbook.add_format({
            'num_format': '$#,##0.00',
            'border': 1
        })
        
        border_format = workbook.add_format({
            'border': 1
        })
        
        # Show progress to user
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_reports = len(reports)
        
        for i, (sheet_name, config) in enumerate(reports.items()):
            progress_text.text(f"Generating {sheet_name}...")
            
            if config['type'] == 'raw':
                result_df = processed_df
            elif config['type'] == 'dashboard':
                result_df = create_dashboard(processed_df, config['dept'], config['exclude'])
            else:
                result_df = create_summary(processed_df, config['dept'], config['exclude'])
            
            result_df.to_excel(writer, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]
            
            # Get the dimensions of the actual data
            last_row = len(result_df.index)
            last_col = len(result_df.columns)
            
            # Format headers
            for col_num, value in enumerate(result_df.columns.values):
                worksheet.write(0, col_num + 1, value, header_format)
            
            # Format data area with borders and currency format if needed
            for row in range(last_row):
                # Format index column
                worksheet.write(row + 1, 0, str(result_df.index[row]), border_format)
                
                # Format data columns
                for col in range(last_col):
                    cell_value = result_df.iloc[row, col]
                    
                    # Check for problematic values based on type
                    if isinstance(cell_value, (int, float)):
                        # For numeric types, check for NaN or Inf
                        if pd.isna(cell_value) or (hasattr(cell_value, 'is_infinite') and cell_value.is_infinite()):
                            cell_value = 0
                    elif cell_value is None or pd.isna(cell_value):
                        # For None or NaN values
                        cell_value = 0 if config['type'] != 'raw' else ''
                    
                    # For string formatting in the raw sheet
                    if config['type'] == 'raw' and not isinstance(cell_value, (int, float, bool)):
                        cell_value = str(cell_value)
                        
                    if config['type'] != 'raw':
                        # For numeric sheets, ensure cell_value is numeric
                        try:
                            if not isinstance(cell_value, (int, float)):
                                cell_value = float(cell_value) if cell_value != '' else 0
                        except (ValueError, TypeError):
                            cell_value = 0
                        worksheet.write(row + 1, col + 1, cell_value, currency_format)
                    else:
                        worksheet.write(row + 1, col + 1, cell_value, border_format)
            
            # Set column widths
            worksheet.set_column(0, 0, 15)
            for col in range(last_col):
                max_length = max(
                    result_df.iloc[:, col].astype(str).apply(len).max() if len(result_df) > 0 else 0,
                    len(str(result_df.columns[col]))
                ) + 2
                worksheet.set_column(col + 1, col + 1, min(max_length, 12))
            
            # Update progress bar
            progress_bar.progress((i + 1) / total_reports)
        
        progress_text.text("Report generation complete!")
    
    if output_file is None:
        # Reset buffer position to the beginning
        output_buffer.seek(0)
        return output_buffer
    else:
        return None

def read_timesheet_data(file):
    """Read timesheet data from a CSV or Excel file"""
    # Determine file type based on extension
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        # Read CSV
        df = pd.read_csv(file)
    elif file_extension in ['xls', 'xlsx']:
        # Read Excel
        df = pd.read_excel(file)
    else:
        st.error(f"Unsupported file format: {file_extension}. Please upload a CSV or Excel file.")
        return None
    
    # Clean numeric values - handle NaN/Inf values immediately
    numeric_columns = ['Wage', 'Regular hours', 'OT hours', 'Double OT hours', 
                       'Holiday hours', 'Regular pay', 'OT pay', 'Double OT pay',
                       'Exception costs', 'Holiday pay', 'Total pay']
    
    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Replace NaN values with 0
            df[col] = df[col].fillna(0)
    
    # Process the DataFrame
    # Convert date columns
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract month name and day of week
    if 'Date' in df.columns:
        df['Month'] = df['Date'].dt.strftime('%B')
        df['DoW'] = df['Date'].dt.strftime('%A')
    
    # Combine First and Last names (if they exist separately)
    if 'First' in df.columns and 'Last' in df.columns and 'Employee' not in df.columns:
        df['Employee'] = df['Last'] + ', ' + df['First']
    
    # Convert time strings to proper format and strip whitespace
    if 'In Time' in df.columns and 'Out Time' in df.columns:
        # Clean and standardize time formats
        df['Clock_In_Time'] = df['In Time'].astype(str).str.strip()
        df['Clock_Out_Time'] = df['Out Time'].astype(str).str.strip()
    elif 'Clock_In_Time' in df.columns and 'Clock_Out_Time' in df.columns:
        # If columns already exist, make sure they're cleaned
        df['Clock_In_Time'] = df['Clock_In_Time'].astype(str).str.strip()
        df['Clock_Out_Time'] = df['Clock_Out_Time'].astype(str).str.strip()
    
    return df

# Main app logic
def main():
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Display file details
        st.write(f"File: {uploaded_file.name}")
        
        try:
            # Read the data
            with st.spinner("Reading file..."):
                df = read_timesheet_data(uploaded_file)
            
            if df is not None:
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Process button
                if st.button("Generate Report"):
                    # Generate the Excel report
                    with st.spinner("Generating report..."):
                        output_buffer = generate_excel_report(df)
                    
                    # Create a download button for the generated report
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"Labor_Report_{timestamp}.xlsx"
                    
                    st.success("Report generated successfully!")
                    st.download_button(
                        label="Download Excel Report",
                        data=output_buffer,
                        file_name=output_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    

        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()