import os
import streamlit as st
from screening import screen_candidates, generate_screening_report, save_report
from scheduling import InterviewScheduler, InterviewSchedule
from interview_feedback import InterviewAnalyzer, PDFTranscriptReader, InterviewFeedback
import tempfile
import shutil
from datetime import datetime, timedelta
import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import asdict
import io

# Set page config
st.set_page_config(
    page_title="HumAINex - AI Interview Tracking System",
    page_icon="🤝",
    layout="wide"
)

# User Authentication Functions
def load_user_database():
    """Load user credentials from Excel file."""
    try:
        # Try to load existing user database
        if os.path.exists("user_database.xlsx"):
            df = pd.read_excel("user_database.xlsx")
            return df
        else:
            # Create default user database if it doesn't exist
            default_users = {
                'Username': [
                    'admin@hr.company.com',
                    'manager@hr.company.com',
                    'john.doe@interviewer.company.com',
                    'jane.smith@interviewer.company.com',
                    'recruiter@hr.company.com'
                ],
                'Password': [
                    'admin123',
                    'manager123',
                    'interviewer123',
                    'interviewer456',
                    'recruiter123'
                ]
            }
            df = pd.DataFrame(default_users)
            df.to_excel("user_database.xlsx", index=False)
            return df
    except Exception as e:
        st.error(f"Error loading user database: {str(e)}")
        return pd.DataFrame(columns=['Username', 'Password'])

def authenticate_user(username, password):
    """Authenticate user credentials."""
    df = load_user_database()
    user_row = df[(df['Username'] == username) & (df['Password'] == password)]
    
    if not user_row.empty:
        # Determine user role based on email domain
        if '@hr.' in username.lower():
            return True, 'hr'
        elif '@interviewer.' in username.lower():
            return True, 'interviewer'
        else:
            return True, 'user'  # Default role
    return False, None

def check_authentication():
    """Check if user is authenticated and return role."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.username = None
    
    return st.session_state.authenticated, st.session_state.get('user_role', None)

def login_page():
    """Display login page."""
    st.title("🔐 HumAINex Login")
    st.subheader("AI-powered Candidate Interview Tracking System")
    
    # Create login form
    with st.container():
        st.markdown("### Please log in to continue")
        
        # Load user database to show available users (for demo purposes)
        df = load_user_database()
        
        # with st.expander("📋 Demo Credentials (Click to view)", expanded=False):
        #     st.markdown("**HR Users (Full Access):**")
        #     hr_users = df[df['Username'].str.contains('@hr.', case=False)]
        #     for _, user in hr_users.iterrows():
        #         st.code(f"Username: {user['Username']}\nPassword: {user['Password']}")
            
        #     st.markdown("**Interviewer Users (Feedback Analysis Only):**")
        #     interviewer_users = df[df['Username'].str.contains('@interviewer.', case=False)]
        #     for _, user in interviewer_users.iterrows():
        #         st.code(f"Username: {user['Username']}\nPassword: {user['Password']}")
        
        # Login form
        with st.form("login_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                username = st.text_input("📧 Username (Email)", placeholder="user@domain.com")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
                submit_button = st.form_submit_button("🚀 Login", type="primary", use_container_width=True)
        
        if submit_button:
            if username and password:
                is_valid, role = authenticate_user(username, password)
                
                if is_valid:
                    st.session_state.authenticated = True
                    st.session_state.user_role = role
                    st.session_state.username = username
                    st.success(f"✅ Login successful! Welcome, {username}")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please check your username and password.")
            else:
                st.warning("⚠️ Please enter both username and password.")

def logout():
    """Handle user logout."""
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.username = None
    st.rerun()

def create_user_management_section():
    """Create user management section for HR users."""
    if st.session_state.get('user_role') == 'hr':
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 User Management")
            
            if st.button("👥 Manage Users"):
                st.session_state.show_user_management = True
            
            if st.button("🔄 Reload User Database"):
                load_user_database()
                st.success("✅ User database reloaded!")

def display_user_management():
    """Display user management interface."""
    st.header("👥 User Management")
    
    df = load_user_database()
    
    # Display current users
    st.subheader("📋 Current Users")
    
    # Color code users by role
    def highlight_role(row):
        if '@hr.' in row['Username'].lower():
            return ['background-color: #d4edda'] * len(row)  # Light green for HR
        elif '@interviewer.' in row['Username'].lower():
            return ['background-color: #d1ecf1'] * len(row)  # Light blue for interviewer
        else:
            return ['background-color: #f8f9fa'] * len(row)  # Light gray for others
    
    styled_df = df.style.apply(highlight_role, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # Legend
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟢 **HR Users** - Full Access")
    with col2:
        st.markdown("🔵 **Interviewer Users** - Feedback Only")
    with col3:
        st.markdown("⚪ **Other Users** - Limited Access")
    
    # Add new user
    st.subheader("➕ Add New User")
    
    with st.form("add_user_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input("📧 Email Address", placeholder="user@domain.com")
            
        with col2:
            new_password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
        
        # Role selection helper
        st.info("💡 **Role Assignment**: Use '@hr.' in email for HR access, '@interviewer.' for interviewer access")
        
        if st.form_submit_button("➕ Add User", type="primary"):
            if new_username and new_password:
                # Check if user already exists
                if new_username in df['Username'].values:
                    st.error("❌ User already exists!")
                else:
                    # Add new user
                    new_user = pd.DataFrame({
                        'Username': [new_username],
                        'Password': [new_password]
                    })
                    
                    updated_df = pd.concat([df, new_user], ignore_index=True)
                    updated_df.to_excel("user_database.xlsx", index=False)
                    
                    # Determine role for display
                    if '@hr.' in new_username.lower():
                        role_display = "HR (Full Access)"
                    elif '@interviewer.' in new_username.lower():
                        role_display = "Interviewer (Feedback Only)"
                    else:
                        role_display = "User (Limited Access)"
                    
                    st.success(f"✅ User added successfully!\n**Email**: {new_username}\n**Role**: {role_display}")
                    st.rerun()
            else:
                st.warning("⚠️ Please fill in all fields.")


def save_uploaded_files(uploaded_files, target_dir: str):
    """Save uploaded files to target directory."""
    os.makedirs(target_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(('.docx', '.pdf', '.txt')):
            file_path = os.path.join(target_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

def screening_tab():
    """Handle the screening functionality."""
    st.header("📋 Candidate Screening")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Sidebar for inputs
        with st.sidebar:
            st.header("1. Job Description")
            jd_file = st.file_uploader(
                "Upload Job Description",
                type=["docx", "pdf", "txt"]
            )
            
            st.header("2. Resumes")
            uploaded_files = st.file_uploader(
                "Upload Candidate Resumes",
                type=["docx", "pdf", "txt"],
                accept_multiple_files=True
            )
            
            # Create a container for the buttons
            with st.container():
                # Create two equal columns
                col1, col2 = st.columns(2)
                
                with col1:
                    process_button = st.button(
                        "Process Resumes",
                        help="Process only the uploaded resumes",
                        use_container_width=True
                    )
                
                with col2:
                    process_db_button = st.button(
                        "Process Database",
                        help="Process all resumes from the database",
                        use_container_width=True
                    )
            
            if process_button or process_db_button:
                if not jd_file:
                    st.error("Please upload a job description first.")
                elif process_button and not uploaded_files:
                    st.error("Please upload at least one resume.")
                else:
                    try:
                        # Save job description
                        jd_path = os.path.join(temp_dir, jd_file.name)
                        save_uploaded_files([jd_file], temp_dir)
                        
                        # Determine which resumes to process
                        if process_button:
                            # Process only uploaded resumes (excluding JD file if it was accidentally included)
                            uploaded_resumes = [f for f in uploaded_files if f.name.lower() != jd_file.name.lower()]
                            if not uploaded_resumes:
                                st.error("No valid resumes found. Please upload at least one resume that is not the job description.")
                                return
                                
                            save_uploaded_files(uploaded_resumes, temp_dir)
                            resume_dir = temp_dir
                            st.session_state['processing_mode'] = 'uploaded'
                        else:
                            # Process all resumes from the database (excluding any file that might be the JD)
                            resume_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resumes')
                            # Store the JD filename to exclude it later
                            st.session_state['jd_filename'] = jd_file.name.lower()
                            st.session_state['processing_mode'] = 'database'
                        
                        # Process resumes
                        with st.spinner("Screening candidates..."):
                            candidates = screen_candidates(
                                resumes_dir=resume_dir,
                                jd_path=jd_path
                            )
                            
                            # Store results in session state
                            report = generate_screening_report(candidates)
                            st.session_state['screening_report'] = report
                            st.session_state['top_candidates'] = candidates  # Store all screened candidates
                            st.session_state['screened_candidates'] = candidates  # Keep for backward compatibility
                            st.session_state['jd_uploaded'] = True
                            st.session_state['resumes_uploaded'] = True
                            st.session_state['processing_complete'] = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.session_state['screening_report'] = None
                        st.session_state['candidates'] = None
                 # Main content area
        # st.header("Candidate Pipeline")
        
        # Display processing mode if set
        if 'processing_mode' in st.session_state:
            mode = st.session_state['processing_mode']
            if mode == 'uploaded':
                st.info("ℹ️ Showing results for uploaded resumes")
            else:
                st.info("ℹ️ Showing results from database")
        
        # Check if both JD and resumes are uploaded and processed
        if not jd_file:
            st.info("👈 Please upload a job description to get started")
        elif not uploaded_files and 'processing_mode' not in st.session_state and 'resumes_uploaded' not in st.session_state:
            st.info("👈 Please upload candidate resumes or click 'Process DB' to use database")
        elif 'screening_report' in st.session_state and st.session_state['screening_report']:
            # Display the pipeline chart if we have screened candidates
            if 'screened_candidates' in st.session_state and st.session_state.screened_candidates:
                # Show pipeline chart
                st.subheader("Recruitment Tracker")
                pipeline_fig = create_pipeline_chart()
                if pipeline_fig:
                    st.plotly_chart(pipeline_fig, use_container_width=True)
                
                # # Show Gantt chart for interview timeline
                # st.subheader("Interview Tracker")
                # gantt_fig = create_gantt_chart()
                # if gantt_fig:
                #     st.plotly_chart(gantt_fig, use_container_width=True)
                
                # # Display the screening report and candidate details
                st.subheader("Top 3 Candidates")
                   
                     
                # Display individual candidate reports using columns for better layout
                for i, candidate in enumerate(st.session_state.screened_candidates[:3], 1):
                    eval_data = candidate.get('evaluation', {})
                    with st.container():
                        st.markdown(f"### {i}. {candidate['name']} (Score: {eval_data.get('score', 0)}/100)")
                        
                        # Use columns to create a two-column layout
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("**Strengths:**")
                            for strength in eval_data.get('strengths', [])[:3]:
                                st.markdown(f"- {strength}")
                        
                        with col2:
                            if eval_data.get('concerns'):
                                st.markdown("**Concerns:**")
                                for concern in eval_data.get('concerns', [])[:2]:
                                    st.markdown(f"- {concern}")
                        
                        st.markdown("---")  # Divider between candidates
                
                # Display full report in expander
                with st.expander("Screening Report of All Candidates"):
                    st.text(st.session_state.get('screening_report', ''))
        elif process_button:
            st.info("⏳ Processing resumes, please wait...")
        elif not jd_file:
            st.info("👈 Please upload a job description to get started")
        else:
            st.info("👈 Please upload selected resume(s) or the entire DB")

def create_gantt_chart():
    """
    Create a Gantt chart showing the interview pipeline timeline.
    X-axis: Weeks (wk1, wk2, etc.)
    Y-axis: Interview events for each candidate
    """
    if 'screened_candidates' not in st.session_state or not st.session_state.screened_candidates:
        return None

    # Get top 3 candidates sorted by score (highest first)
    top_candidates = sorted(
        st.session_state.screened_candidates,
        key=lambda x: x.get('evaluation', {}).get('score', 0),
        reverse=True
    )[:3]

    # Define interview stages and their durations (in weeks)
    interview_stages = [
        {'name': 'L1-request-sent', 'duration': 0.5, 'color': '#636EFA'},
        {'name': 'L1-request-accepted', 'duration': 0.5, 'color': '#00CC96'},
        {'name': 'L1-invite-sent', 'duration': 0.5, 'color': '#AB63FA'},
        {'name': 'L2-request-sent', 'duration': 0.5, 'color': '#FFA15A'},
        {'name': 'L2-request-accepted', 'duration': 0.5, 'color': '#19D3F3'},
        {'name': 'L2-invite-sent', 'duration': 0.5, 'color': '#FF6692'},
        {'name': 'L3-request-sent', 'duration': 0.5, 'color': '#B6E880'},
        {'name': 'L3-request-accepted', 'duration': 0.5, 'color': '#FF97FF'},
        {'name': 'L3-invite-sent', 'duration': 0.5, 'color': '#FECB52'}
    ]

    # Create figure
    fig = go.Figure()

    # Add traces for each candidate
    for i, candidate in enumerate(top_candidates):
        candidate_name = candidate['name']
        start_week = 0.5  # Start from week 0.5
        
        for stage in interview_stages:
            end_week = start_week + stage['duration']
            
            # Add a bar for this stage
            fig.add_trace(go.Bar(
                x=[end_week - start_week],  # Width of the bar
                y=[f"{candidate_name} ({candidate.get('evaluation', {}).get('score', 0)}/100)"],
                name=stage['name'],
                orientation='h',
                marker=dict(
                    color=stage['color'],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                ),
                base=start_week,
                hoverinfo='text',
                hovertext=f"{stage['name']}: Week {start_week:.1f}-{end_week:.1f}",
                showlegend=False
            ))
            
            # Add stage label in the middle of the bar
            fig.add_annotation(
                x=start_week + (end_week - start_week) / 2,
                y=f"{candidate_name} ({candidate.get('evaluation', {}).get('score', 0)}/100)",
                text=stage['name'],
                showarrow=False,
                font=dict(size=8, color='white'),
                textangle=0,
                xanchor='center',
                yanchor='middle'
            )
            
            start_week = end_week  # Move to next stage

    # Update layout for Gantt chart
    fig.update_layout(
        title='Interview Timeline (Weeks)',
        title_x=0.5,
        xaxis=dict(
            title='Weeks',
            showgrid=True,
            gridcolor='lightgray',
            tickmode='array',
            tickvals=list(range(0, 6)),
            ticktext=[f'wk{i}' for i in range(0, 6)],
            range=[0, 5]  # Show up to 5 weeks
        ),
        yaxis=dict(
            title='Candidates',
            showgrid=True,
            gridcolor='lightgray',
            autorange='reversed'  # Highest score at top
        ),
        barmode='stack',
        height=300 + len(top_candidates) * 60,
        margin=dict(l=200, r=20, t=60, b=30, pad=4),
        plot_bgcolor='white',
        showlegend=False
    )
    
    # Add vertical grid lines for each week
    for week in range(6):
        fig.add_vline(
            x=week, 
            line_width=1, 
            line_dash="dash", 
            line_color="gray"
        )
    
    return fig

def create_pipeline_chart():
    # Define recruitment stages and their icons
    stages = ['Ready for Evaluation', 'L1 Cleared', 'L2 Cleared', 
              'L3 Cleared', 'Offered', 'Rejected', 'On Hold']
    icons = ['👶', '🧒', '👦', '👨', '👨\u200d💼', '❌', '⏸️']
    
    # Only create chart if we have screened candidates in session state
    if 'screened_candidates' not in st.session_state or not st.session_state.screened_candidates:
        return None
    
    # Get top candidates sorted by score (highest first)
    top_candidates = sorted(
        st.session_state.screened_candidates,
        key=lambda x: x.get('evaluation', {}).get('score', 0),
        reverse=True
    )[:3]  # Take top 3
    
    # Sort candidates by score in descending order for Y-axis (highest on top)
    top_candidates = sorted(
        top_candidates,
        key=lambda x: x.get('evaluation', {}).get('score', 0),
        reverse=True
    )
    
    # Create candidate data with initial 'Ready for Evaluation' stage and store scores
    candidate_data = {}
    for candidate in top_candidates:
        candidate_name = candidate['name']
        score = candidate.get('evaluation', {}).get('score', 0)
        candidate_data[candidate_name] = {
            'stage': 'Ready for Evaluation',
            'score': score
        }
    
    # Create figure
    fig = go.Figure()
    
    # Create a matrix to track which candidate is in which stage
    candidates = list(candidate_data.keys())
    data = []
    
    for i, stage in enumerate(stages):
        # Get all candidates in this stage
        stage_candidates = [cand for cand, data in candidate_data.items() 
                          if data.get('stage') == stage]
        # Create a bar for each candidate in this stage
        for candidate in stage_candidates:
            score = candidate_data[candidate].get('score', 0)
            data.append({
                'candidate': candidate,
                'stage': stage,
                'x': i,  # stage index on x-axis
                'y': candidates.index(candidate),  # candidate index on y-axis
                'score': score
            })
    
    # Add scatter plot points for each candidate at their stage
    for item in data:
        fig.add_trace(go.Scatter(
            x=[item['x']],
            y=[item['y']],
            mode='markers+text',
            name=item['stage'],
            marker=dict(
                size=20,
                symbol='circle',
                color='#1f77b4',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=icons[stages.index(item['stage'])],
            textposition='middle center',
            textfont=dict(size=14),
            hovertemplate=f"<b>{item['candidate']}</b><br>" +
                         f"Stage: {item['stage']}<br>" +
                         (f"Screening Score: {item['score']}/100" if 'score' in item else '') +
                         "<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        # title='Recruitment Tracker',
        xaxis=dict(
            tickmode='array',
            ticktext=[f"<span style='font-size: 24px;'>{icon}</span><br>{'<br>'.join(stage.split())}" for icon, stage in zip(icons, stages)],
            tickvals=list(range(len(stages))),
            tickangle=0,
            title='',
            showgrid=False,
            range=[-0.5, len(stages) - 0.5],
            fixedrange=True,
            showticklabels=True,
            tickfont=dict(size=14, family='Arial'),
            ticklen=10,
            tickwidth=1,
            tickcolor='rgba(0,0,0,0.1)',
            tickformat='<br>',
            ticklabelposition='outside',
            ticklabeloverflow='allow',
            ticklabelstep=1,
            ticklabelmode='period',
            automargin=True,
            side='bottom',
            constrain='domain',
            constraintoward='center',
            dividerwidth=1,
            dividercolor='lightgray',
            showdividers=True
        ),
        yaxis=dict(
            tickmode='array',
            ticktext=[f"{candidate['name']}" for candidate in top_candidates],
            tickvals=list(range(len(top_candidates))),
            title='',
            showgrid=False,
            fixedrange=True,
            tickfont=dict(size=12),
            automargin=True,
            autorange='reversed'  # This ensures highest score is at the top
        ),
        showlegend=False,
        template='plotly_white',
        height=200 + len(candidates) * 40,
        margin=dict(l=10, r=10, t=80, b=150, pad=10),
        plot_bgcolor='white',
        xaxis_showgrid=False,
        yaxis_showgrid=True
    )
    
    # Add some padding to ensure all x-axis labels are visible
    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey')
    
    return fig


def create_gantt_chart():
    """
    Create a Gantt chart showing the interview pipeline timeline.
    X-axis: Weeks (wk1, wk2, etc.)
    Y-axis: Interview events for each candidate
    """
    if 'screened_candidates' not in st.session_state or not st.session_state.screened_candidates:
        return None

    # Get top 3 candidates sorted by score (highest first)
    top_candidates = sorted(
        st.session_state.screened_candidates,
        key=lambda x: x.get('evaluation', {}).get('score', 0),
        reverse=True
    )[:3]

    # Define interview stages and their durations (in weeks)
    interview_stages = [
        {'name': 'L1-request-sent', 'duration': 0.5, 'color': '#636EFA'},
        {'name': 'L1-request-accepted', 'duration': 0.5, 'color': '#636EFb'},
        {'name': 'L1-invite-sent', 'duration': 0.5, 'color': '#636EFc'},
        {'name': 'L2-request-sent', 'duration': 0.5, 'color': '#FFA15A'},
        {'name': 'L2-request-accepted', 'duration': 0.5, 'color': '#FFA15b'},
        {'name': 'L2-invite-sent', 'duration': 0.5, 'color': '#FFA15c'},
        {'name': 'L3-request-sent', 'duration': 0.5, 'color': '#B6E880'},
        {'name': 'L3-request-accepted', 'duration': 0.5, 'color': '#B6E88b'},
        {'name': 'L3-invite-sent', 'duration': 0.5, 'color': '#B6E88c'}
    ]

    # Create figure
    fig = go.Figure()

    # Add traces for each candidate
    for i, candidate in enumerate(top_candidates):
        candidate_name = candidate['name']
        start_week = 0.5  # Start from week 0.5
        
        for stage in interview_stages:
            end_week = start_week + stage['duration']
            
            # Add a bar for this stage
            fig.add_trace(go.Bar(
                x=[end_week - start_week],  # Width of the bar
                y=[f"{candidate_name} ({candidate.get('evaluation', {}).get('score', 0)}/100)"],
                name=stage['name'],
                orientation='h',
                marker=dict(
                    color=stage['color'],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                ),
                base=start_week,
                hoverinfo='text',
                hovertext=f"{stage['name']}: Week {start_week:.1f}-{end_week:.1f}",
                showlegend=False
            ))
            
            # Add stage label in the middle of the bar
            fig.add_annotation(
                x=start_week + (end_week - start_week) / 2,
                y=f"{candidate_name} ({candidate.get('evaluation', {}).get('score', 0)}/100)",
                text=stage['name'],
                showarrow=False,
                font=dict(size=8, color='white'),
                textangle=0,
                xanchor='center',
                yanchor='middle'
            )
            
            start_week = end_week  # Move to next stage

    # Update layout for Gantt chart
    fig.update_layout(
        title='Interview Timeline (Weeks)',
        title_x=0.5,
        xaxis=dict(
            title='Weeks',
            showgrid=True,
            gridcolor='lightgray',
            tickmode='array',
            tickvals=list(range(0, 6)),
            ticktext=[f'wk{i}' for i in range(0, 6)],
            range=[0, 5]  # Show up to 5 weeks
        ),
        yaxis=dict(
            title='Candidates',
            showgrid=True,
            gridcolor='lightgray',
            autorange='reversed'  # Highest score at top
        ),
        barmode='stack',
        height=300 + len(top_candidates) * 60,
        margin=dict(l=200, r=20, t=60, b=30, pad=4),
        plot_bgcolor='white',
        showlegend=False
    )
    
    # Add vertical grid lines for each week
    for week in range(6):
        fig.add_vline(
            x=week, 
            line_width=1, 
            line_dash="dash", 
            line_color="gray"
        )
    
    return fig

def create_pipeline_chart():
    # Define recruitment stages and their icons
    stages = ['Ready for Evaluation', 'L1 Cleared', 'L2 Cleared', 
              'L3 Cleared', 'Offered', 'Rejected', 'On Hold']
    icons = ['👶', '🧒', '👦', '👨', '👨\u200d💼', '❌', '⏸️']
    
    # Only create chart if we have screened candidates in session state
    if 'screened_candidates' not in st.session_state or not st.session_state.screened_candidates:
        return None
    
    # Get top candidates sorted by score (highest first)
    top_candidates = sorted(
        st.session_state.screened_candidates,
        key=lambda x: x.get('evaluation', {}).get('score', 0),
        reverse=True
    )[:3]  # Take top 3
    
    # Sort candidates by score in descending order for Y-axis (highest on top)
    top_candidates = sorted(
        top_candidates,
        key=lambda x: x.get('evaluation', {}).get('score', 0),
        reverse=True
    )
    
    # Create candidate data with initial 'Ready for Evaluation' stage and store scores
    candidate_data = {}
    for candidate in top_candidates:
        candidate_name = candidate['name']
        score = candidate.get('evaluation', {}).get('score', 0)
        candidate_data[candidate_name] = {
            'stage': 'Ready for Evaluation',
            'score': score
        }
    
    # Create figure
    fig = go.Figure()
    
    # Create a matrix to track which candidate is in which stage
    candidates = list(candidate_data.keys())
    data = []
    
    for i, stage in enumerate(stages):
        # Get all candidates in this stage
        stage_candidates = [cand for cand, data in candidate_data.items() 
                          if data.get('stage') == stage]
        # Create a bar for each candidate in this stage
        for candidate in stage_candidates:
            score = candidate_data[candidate].get('score', 0)
            data.append({
                'candidate': candidate,
                'stage': stage,
                'x': i,  # stage index on x-axis
                'y': candidates.index(candidate),  # candidate index on y-axis
                'score': score
            })
    
    # Add scatter plot points for each candidate at their stage
    for item in data:
        fig.add_trace(go.Scatter(
            x=[item['x']],
            y=[item['y']],
            mode='markers+text',
            name=item['stage'],
            marker=dict(
                size=20,
                symbol='circle',
                color='#1f77b4',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=icons[stages.index(item['stage'])],
            textposition='middle center',
            textfont=dict(size=14),
            hovertemplate=f"<b>{item['candidate']}</b><br>" +
                         f"Stage: {item['stage']}<br>" +
                         (f"Screening Score: {item['score']}/100" if 'score' in item else '') +
                         "<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title='',
        xaxis=dict(
            tickmode='array',
            ticktext=[f"<span style='font-size: 24px;'>{icon}</span><br>{'<br>'.join(stage.split())}" for icon, stage in zip(icons, stages)],
            tickvals=list(range(len(stages))),
            tickangle=0,
            title='',
            showgrid=False,
            range=[-0.5, len(stages) - 0.5],
            fixedrange=True,
            showticklabels=True,
            tickfont=dict(size=14, family='Arial'),
            ticklen=10,
            tickwidth=1,
            tickcolor='rgba(0,0,0,0.1)',
            tickformat='<br>',
            ticklabelposition='outside',
            ticklabeloverflow='allow',
            ticklabelstep=1,
            ticklabelmode='period',
            automargin=True,
            side='bottom',
            constrain='domain',
            constraintoward='center',
            dividerwidth=1,
            dividercolor='lightgray',
            showdividers=True
        ),
        yaxis=dict(
            tickmode='array',
            ticktext=[f"{candidate['name']}" for candidate in top_candidates],
            tickvals=list(range(len(top_candidates))),
            title='',
            showgrid=False,
            fixedrange=True,
            tickfont=dict(size=12),
            automargin=True,
            autorange='reversed'  # This ensures highest score is at the top
        ),
        showlegend=False,
        template='plotly_white',
        height=200 + len(candidates) * 40,
        margin=dict(l=10, r=10, t=80, b=150, pad=10),
        plot_bgcolor='white',
        xaxis_showgrid=False,
        yaxis_showgrid=True
    )
    
    # Add some padding to ensure all x-axis labels are visible
    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey')
    
    return fig

    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Sidebar for inputs
        with st.sidebar:
            st.header("1. Job Description")
            jd_file = st.file_uploader(
                "Upload Job Description",
                type=["docx", "pdf", "txt"],
                key="jd_screening"
            )
            
            st.header("2. Upload Resumes")
            uploaded_files = st.file_uploader(
                "Upload Resumes",
                type=["docx", "pdf", "txt"],
                accept_multiple_files=True,
                key="resumes_screening"
            )
            
            use_sample = st.checkbox("Or use sample resumes", key="sample_screening")
            if use_sample:
                sample_dir = "resumes"  # Default sample directory
                if os.path.exists(sample_dir):
                    st.info(f"Using {len(os.listdir(sample_dir))} sample resumes")
        
        # Main content
        if jd_file and (uploaded_files or (use_sample and os.path.exists("resumes"))):
            # Save job description
            jd_path = os.path.join(temp_dir, jd_file.name)
            with open(jd_path, "wb") as f:
                f.write(jd_file.getbuffer())
            
            # Process resumes
            resume_dir = os.path.join(temp_dir, "resumes")
            os.makedirs(resume_dir, exist_ok=True)
            
            # Count total resumes for tracking
            total_resumes = 0
            if uploaded_files:
                save_uploaded_files(uploaded_files, resume_dir)
                total_resumes = len(uploaded_files)
            elif os.path.exists("resumes"):
                # Copy sample resumes
                resume_files = [f for f in os.listdir("resumes") if f.lower().endswith(('.docx', '.pdf', '.txt'))]
                for file in resume_files:
                    shutil.copy(os.path.join("resumes", file), resume_dir)
                total_resumes = len(resume_files)
            
            if st.button("🚀 Start Screening", type="primary"):
                with st.spinner("Screening candidates..."):
                    try:
                        top_candidates = screen_candidates(resumes_dir=resume_dir, jd_path=jd_path)
                        if top_candidates:
                            # Store results in session state for use in scheduling tab
                            st.session_state.top_candidates = top_candidates
                            st.session_state.jd_file_name = jd_file.name
                            st.session_state.screening_completed_at = datetime.now()
                            # Store total resumes count for dashboard analytics
                            st.session_state.total_resumes_screened = total_resumes
                            st.session_state.selected_candidates_count = len(top_candidates)
                            
                            report = generate_screening_report(top_candidates)
                            report_path = save_report(report, "screening_results/screening_report.txt")
                            
                            st.success("✅ Screening completed!")
                            st.info(f"📊 Screened {total_resumes} resumes, selected {len(top_candidates)} top candidates")
                            st.subheader("Top Candidates")
                            
                            for i, candidate in enumerate(top_candidates, 1):
                                eval_data = candidate.get('evaluation', {})
                                with st.expander(f"{i}. {candidate['name']} (Score: {eval_data.get('score', 0)}/100)"):
                                    st.markdown("**Strengths:**")
                                    for strength in eval_data.get('strengths', [])[:3]:
                                        st.markdown(f"- {strength}")
                                    
                                    if 'concerns' in eval_data and eval_data['concerns']:
                                        st.markdown("\n**Considerations:**")
                                        for concern in eval_data.get('concerns', [])[:2]:
                                            st.markdown(f"- {concern}")
                            
                            # Download report
                            with open(report_path, "r") as f:
                                st.download_button(
                                    "📥 Download Full Report",
                                    f.read(),
                                    "screening_report.txt",
                                    "text/plain"
                                )
                            
                            st.info("💡 You can now schedule interviews in the 'Schedule Interviews' tab!")
                            
                    except Exception as e:
                        st.error(f"Error during screening: {str(e)}")
        elif not jd_file:
            st.info("👈 Please upload a job description to get started")
        else:
            st.info("👈 Please upload resumes or use sample resumes")

def scheduling_tab():
    """Handle the interview scheduling functionality."""
    st.header("📅 Interview Scheduling")
    
    # Check if candidates have been screened
    if not st.session_state.get('top_candidates'):
        st.warning("⚠️ Please screen candidates first in the 'Screening' tab before scheduling interviews.")
        return
    
    top_candidates = st.session_state.top_candidates
    
    st.success(f"✅ {len(top_candidates)} candidates available for scheduling")
    
    # Scheduling configuration
    st.subheader("Interview Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        interviewer_name = st.text_input("Interviewer Name", value="Hiring Manager")
        interviewer_email = st.text_input("Interviewer Email", placeholder="interviewer@company.com")
        common_positions = [
            "Software Engineer",
            "Senior Software Engineer",
            "Frontend Developer",
            "Backend Developer",
            "Full Stack Developer",
            "DevOps Engineer",
            "Data Scientist",
            "Machine Learning Engineer",
            "QA Engineer",
            "Product Manager",
            "Project Manager",
            "UI/UX Designer",
            "Systems Administrator",
            "Cloud Architect",
            "Security Engineer"
        ]
        
        # If there's a JD filename, use it as the default position
        default_position = st.session_state.get('jd_file_name', '').replace('.docx', '').replace('_JD', '').replace('_', ' ')
        if default_position:
            common_positions = [default_position] + [p for p in common_positions if p.lower() != default_position.lower()]
        
        position = st.selectbox(
            "Position",
            options=[""] + common_positions,
            format_func=lambda x: 'Type or select...' if x == "" else x,
            index=0 if not default_position else common_positions.index(default_position) + 1
        )
        
        # Allow custom input if needed
        if position == "":
            position = st.text_input("Or enter custom position:", key="custom_position")
    
    with col2:
        interview_type = st.selectbox("Interview Type", ["L1-Round", "L2-Round", "L3-Round", "General"])
        duration = st.selectbox("Duration (minutes)", [30, 45, 60], index=2)
        timezone = st.selectbox("Timezone", ["Asia/Kolkata", "UTC", "US/Eastern", "US/Pacific", "Europe/London"], index=0)
    
    st.subheader("Schedule Interviews")
    
    # Show candidates and allow individual scheduling
    for i, candidate in enumerate(top_candidates):
        eval_data = candidate.get('evaluation', {})
        
        with st.expander(f"{i+1}. {candidate['name']} (Score: {eval_data.get('score', 0)}/100)", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                # Date picker
                min_date = datetime.now().date() + timedelta(days=1)
                interview_date = st.date_input(
                    "Interview Date",
                    value=min_date + timedelta(days=i),
                    min_value=min_date,
                    key=f"date_{i}"
                )
            
            with col2:
                # Time picker
                interview_time = st.time_input(
                    "Interview Time",
                    value=datetime.strptime("10:00", "%H:%M").time(),
                    key=f"time_{i}"
                )
            
            with col3:
                # Schedule button
                if st.button(f"📅 Schedule", key=f"schedule_{i}", type="primary"):
                    if not interviewer_email:
                        st.error("Please enter interviewer email")
                    else:
                        # Combine date and time
                        interview_datetime = datetime.combine(interview_date, interview_time)
                        
                        # Debug: Print candidate object structure
                        print(f"[DEBUG] Candidate object: {candidate}")
                        
                        # Get candidate name safely
                        candidate_name = candidate.get('name', 'Unknown Candidate')
                        
                        # Generate a placeholder email if not available
                        candidate_email = "vedravitej@gmail.com" 
                        #candidate.get('candidate_email') or candidate.get('email')
                        if not candidate_email:
                            # Create a placeholder email using candidate's name
                            safe_name = ''.join(c for c in candidate_name.lower() if c.isalnum() or c.isspace())
                            candidate_email = f"{safe_name.replace(' ', '.')}@example.com"
                            print(f"[DEBUG] Generated placeholder email: {candidate_email}")
                        
                        # Create interview schedule
                        try:
                            schedule = InterviewSchedule(
                                candidate_name=candidate_name,
                                candidate_email=candidate_email,
                                interviewer_name=interviewer_name,
                                interviewer_email=interviewer_email,
                                position=position,
                                interview_type=interview_type,
                                scheduled_time=interview_datetime,
                                duration_minutes=duration,
                                meeting_link=f"https://teams.microsoft.com/xyz-{hash(candidate_name) % 10000}-abc",
                                timezone=timezone
                            )
                            print("[DEBUG] Successfully created schedule object")
                        except Exception as e:
                            print(f"[ERROR] Failed to create schedule: {e}")
                            raise
                        
                        try:
                            # Initialize scheduler
                            scheduler = InterviewScheduler()
                            success, message = scheduler.schedule_interview(schedule)
                            
                            if success:
                                st.success(f"✅ Interview scheduled for {candidate['name']} on {interview_date} at {interview_time}")
                                
                                # Store scheduled interview info
                                if 'scheduled_interviews' not in st.session_state:
                                    st.session_state.scheduled_interviews = []
                                
                                interview_info = {
                                    'candidate_name': candidate['name'],
                                    'candidate_email': candidate['email'],
                                    'interview_datetime': interview_datetime.isoformat(),
                                    'position': position,
                                    'interview_type': interview_type,
                                    'meeting_link': schedule.meeting_link,
                                    'duration': duration,
                                    'interviewer_name': interviewer_name,
                                    'scheduled_at': datetime.now().isoformat(),
                                    'status': 'scheduled'
                                }
                                st.session_state.scheduled_interviews.append(interview_info)
                                
                                # Update session state to reflect the change immediately
                                st.session_state.interviews_scheduled_count = len(st.session_state.scheduled_interviews)
                                
                                # Schedule reminder
                                scheduler.send_reminder(schedule, hours_before=24)
                                st.info("📧 Email notifications sent to candidate and interviewer")
                                
                                # Force refresh of the dashboard data
                                st.rerun()
                                
                            else:
                                st.error(f"❌ Failed to schedule interview: {message}")
                                
                        except Exception as e:
                            # st.error(f"❌ Error scheduling interview: {str(e)}")
                            print(f"❌ Error scheduling interview: {str(e)}")
                            
            # Show candidate details
            st.markdown("**Key Strengths:**")
            for strength in eval_data.get('strengths', [])[:2]:
                st.markdown(f"- {strength}")
    
    # Bulk scheduling option
    st.subheader("Bulk Schedule All Candidates")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() + timedelta(days=1),
            min_value=datetime.now().date() + timedelta(days=1)
        )
        start_time = st.time_input(
            "Start Time",
            value=datetime.strptime("10:00", "%H:%M").time()
        )
    
    with col2:
        interval_hours = st.selectbox("Interview Interval", [2, 3, 4, 24], index=1, 
                                     format_func=lambda x: f"{x} hours" if x < 24 else "1 day")
    
    if st.button("📅 Schedule All Interviews", type="primary"):
        if not interviewer_email:
            st.error("Please enter interviewer email")
        else:
            with st.spinner("Scheduling interviews for all candidates..."):
                try:
                    scheduler = InterviewScheduler()
                    scheduled_count = 0
                    
                    # Initialize session state if not exists
                    if 'scheduled_interviews' not in st.session_state:
                        st.session_state.scheduled_interviews = []
                    
                    # Initialize scheduled count in session state if not exists
                    if 'interviews_scheduled_count' not in st.session_state:
                        st.session_state.interviews_scheduled_count = 0
                    
                    # Make a copy of the current scheduled interviews
                    current_scheduled = list(st.session_state.scheduled_interviews)
                    scheduled_names = {interview['candidate_name'] for interview in current_scheduled}
                    
                    # Reset scheduled count at the start of scheduling
                    scheduled_count = 0
                    
                    # Only process candidates that aren't already scheduled
                    candidates_to_schedule = [c for c in top_candidates if c['name'] not in scheduled_names]
                    
                    if not candidates_to_schedule:
                        st.warning("⚠️ All candidates already have scheduled interviews!")
                    else:
                        st.info(f"ℹ️ Found {len(candidates_to_schedule)} candidates to schedule out of {len(top_candidates)} total")
                        
                        new_interviews = []
                        
                        for i, candidate in enumerate(candidates_to_schedule):
                            try:
                                # Calculate interview time
                                if interval_hours == 24:
                                    interview_datetime = datetime.combine(start_date + timedelta(days=i), start_time)
                                else:
                                    interview_datetime = datetime.combine(start_date, start_time) + timedelta(hours=i * interval_hours)
                                
                                # Create schedule
                                schedule = InterviewSchedule(
                                    candidate_name=candidate['name'],
                                    candidate_email=candidate.get('email', f"{candidate['name'].lower().replace(' ', '.')}@example.com"),
                                    interviewer_name=interviewer_name,
                                    interviewer_email=interviewer_email,
                                    position=position,
                                    interview_type=interview_type,
                                    scheduled_time=interview_datetime,
                                    duration_minutes=duration,
                                    meeting_link=f"https://meet.google.com/xyz-{str(abs(hash(candidate['name']))) % 10000}-abc",
                                    timezone=timezone
                                )
                                
                                success, message = scheduler.schedule_interview(schedule)
                                if success:
                                    scheduled_count += 1
                                    
                                    # Create interview info
                                    interview_info = {
                                        'candidate_name': candidate['name'],
                                        'candidate_email': candidate.get('email', f"{candidate['name'].lower().replace(' ', '.')}@example.com"),
                                        'interview_datetime': interview_datetime.isoformat(),
                                        'position': position,
                                        'interview_type': interview_type,
                                        'meeting_link': schedule.meeting_link,
                                        'duration': duration,
                                        'interviewer_name': interviewer_name,
                                        'scheduled_at': datetime.now().isoformat(),
                                        'status': 'scheduled'
                                    }
                                    new_interviews.append(interview_info)
                                    
                                    # Don't block on sending reminders
                                    try:
                                        scheduler.send_reminder(schedule, hours_before=24)
                                    except Exception as e:
                                        print(f"Warning: Failed to schedule reminder: {e}")
                                        
                                    print(f"✅ Scheduled interview for {candidate['name']} at {interview_datetime}")
                                    
                            except Exception as e:
                                print(f"Error scheduling interview for {candidate.get('name', 'unknown')}: {e}")
                        
                        # Update session state with all new interviews at once
                        if new_interviews:
                            st.session_state.scheduled_interviews = current_scheduled + new_interviews
                            # Force session state update
                            st.session_state['_scheduled_interviews_updated'] = True
                    
                    # Update session state with the new count
                    st.session_state.interviews_scheduled_count = len(st.session_state.scheduled_interviews)
                    
                    if scheduled_count > 0:
                        # Force update the session state
                        st.session_state['_scheduled_count_updated'] = True
                        st.success(f"✅ Successfully scheduled {scheduled_count} interviews!")
                        st.info("📧 Email notifications sent to all candidates and interviewer")
                        # Force a rerun to update the UI with the new count
                        st.rerun()
                    else:
                        st.error("❌ Failed to schedule any interviews")
                        
                except Exception as e:
                    st.error(f"❌ Error during bulk scheduling: {str(e)}")
    
    # Show scheduled interviews
    if st.session_state.get('scheduled_interviews'):
        st.subheader("📋 Scheduled Interviews")
        
        for interview in st.session_state.scheduled_interviews:
            interview_dt = datetime.fromisoformat(interview['interview_datetime'])
            st.info(f"**{interview['candidate_name']}** - {interview_dt.strftime('%Y-%m-%d %I:%M %p')} | {interview['interview_type']} | [Meeting Link]({interview['meeting_link']})")

def feedback_analysis_tab():
    """Handle the interview feedback analysis functionality."""
    st.header("🎯 Interview Feedback Analysis")
    
    # Initialize session state for feedback analysis
    if 'analyzed_feedbacks' not in st.session_state:
        st.session_state.analyzed_feedbacks = []
    
    # Configuration section
    st.subheader("⚙️ Configuration")
    
    # Get API key using the same method as screening.py
    try:
        from config import get_api_key
        api_key = get_api_key()
        st.success("✅ Using OpenAI API key from configuration")
    except Exception as e:
        st.error(f"❌ Error loading API key: {str(e)}")
        api_key = ""
    
    model_choice = "gpt-4o-mini"
    
    st.subheader("📄 Upload Interview Transcripts")
    
    # File upload section
    uploaded_transcripts = st.file_uploader(
        "Upload PDF Transcripts",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF files containing interview transcripts. The AI will automatically extract candidate name, role, and interview round from the transcript content."
    )
    
    if uploaded_transcripts:
        st.success(f"✅ {len(uploaded_transcripts)} transcript(s) uploaded")
        # st.info("🤖 AI will automatically extract candidate name, role, and interview round from each transcript")
        
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key above")
        else:
            # Individual analysis section
            st.subheader("🔍 Individual Analysis")
            
            # Show uploaded files with auto-extraction preview
            for i, uploaded_file in enumerate(uploaded_transcripts):
                with st.expander(f"📋 {uploaded_file.name}", expanded=False):
                    st.write("**File:** " + uploaded_file.name)
                    st.write("**AI will extract:** Candidate Name, Role, Interview Round, Interviewer Name")
                    st.info("💡 The AI will analyze the transcript content to automatically identify all metadata")
                    
                    if st.button(f"🎯 Analyze Transcript", key=f"analyze_{i}", type="primary", use_container_width=True):
                        if not api_key:
                            st.error("❌ Please set the OPENAI_API_KEY environment variable to analyze transcripts")
                        else:
                            with st.spinner(f"Extracting metadata and analyzing {uploaded_file.name}..."):
                                tmp_path = None
                                try:
                                    # Save uploaded file temporarily
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                        tmp_file.write(uploaded_file.getvalue())
                                        tmp_path = tmp_file.name
                                    
                                    # Initialize analyzer with error handling
                                    try:
                                        analyzer = InterviewAnalyzer(api_key, model_choice)
                                    except Exception as e:
                                        st.error(f"❌ Failed to initialize analyzer: {str(e)}")
                                        if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                                            os.unlink(tmp_path)
                                        st.stop()
                                    
                                    # Analyze the transcript with automatic metadata extraction
                                    feedback = analyzer.analyze_pdf_transcript(
                                        pdf_path=tmp_path
                                    )
                                    
                                    # Store feedback
                                    st.session_state.analyzed_feedbacks.append(feedback)
                                    
                                    # Clean up temp file
                                    os.unlink(tmp_path)
                                    
                                    st.success("✅ Analysis completed!")
                                    
                                    # Display extracted metadata in proper columns with better spacing
                                    st.markdown("### 🔍 Extracted Information")
                                    
                                    # Use a single column layout for better readability
                                    st.markdown(f"""
                                    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                                        <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                                            <div style='flex: 1; padding: 5px;'><strong>👤 Candidate:</strong> {feedback.candidate_name}</div>
                                            <div style='flex: 1; padding: 5px;'><strong>💼 Role:</strong> {feedback.role}</div>
                                        </div>
                                        <div style='display: flex; justify-content: space-between;'>
                                            <div style='flex: 1; padding: 5px;'><strong>🔄 Interview Round:</strong> {feedback.interview_round}</div>
                                            <div style='flex: 1; padding: 5px;'><strong>👨‍💼 Interviewer:</strong> {feedback.interviewer_name}</div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Add custom CSS for full-width layout
                                    st.markdown("""
                                    <style>
                                        .stApp {
                                            max-width: 100% !important;
                                            padding: 20px !important;
                                        }
                                        .main .block-container {
                                            max-width: 100% !important;
                                            padding: 2rem 2rem !important;
                                        }
                                        .full-width {
                                            width: 100% !important;
                                            max-width: 100% !important;
                                        }
                                        .metric-container {
                                            display: flex;
                                            flex-wrap: wrap;
                                            gap: 10px;
                                            margin-bottom: 20px;
                                            width: 100% !important;
                                        }
                                        .metric-box {
                                            flex: 1 1 200px;
                                            min-width: 200px;
                                            padding: 15px;
                                            background: #f8f9fa;
                                            border-radius: 8px;
                                            box-sizing: border-box;
                                        }
                                        .feedback-section {
                                            background: #f8f9fa;
                                            padding: 20px;
                                            border-radius: 8px;
                                            margin: 15px 0;
                                            width: 100% !important;
                                        }
                                        h3 {
                                            margin: 0;
                                            font-size: 1.5rem;
                                        }
                                    </style>
                                    """, unsafe_allow_html=True)
                                    
                                    # Display analysis results in a full-width container
                                    st.markdown("### 📊 Analysis Results")
                                    st.markdown(f"""
                                    <div class="full-width">
                                        <div class="metric-container">
                                            <div class="metric-box">
                                                <div>📈 Overall Rating</div>
                                                <h3>{feedback.overall_rating}/5</h3>
                                            </div>
                                            <div class="metric-box">
                                                <div>🔧 Technical Skills</div>
                                                <h3>{feedback.technical_skills}/5</h3>
                                            </div>
                                            <div class="metric-box">
                                                <div>💬 Communication</div>
                                                <h3>{feedback.communication}/5</h3>
                                            </div>
                                            <div class="metric-box">
                                                <div>🧩 Problem Solving</div>
                                                <h3>{feedback.problem_solving}/5</h3>
                                            </div>
                                            <div class="metric-box">
                                                <div>🤝 Cultural Fit</div>
                                                <h3>{feedback.cultural_fit}/5</h3>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Recommendation
                                    st.markdown("### 🎯 Recommendation")
                                    if "hire" in feedback.recommendation.lower() and "no" not in feedback.recommendation.lower():
                                        st.success(feedback.recommendation)
                                    elif "not" in feedback.recommendation.lower() or "no hire" in feedback.recommendation.lower():
                                        st.error(feedback.recommendation)
                                    else:
                                        st.warning(feedback.recommendation)
                                    
                                    # Full width sections for feedback
                                    st.markdown("---")
                                    
                                    def format_as_bullets(content):
                                        if isinstance(content, list):
                                            items = [str(item).strip() for item in content if str(item).strip()]
                                        else:
                                            items = [s.strip() for s in str(content).split('\n') if s.strip()]
                                        return "<div class='feedback-section'><ul>" + "".join([f"<li>{item}</li>" for item in items]) + "</ul></div>"
                                    
                                    st.markdown("### 💪 Strengths")
                                    st.markdown(format_as_bullets(feedback.strengths), unsafe_allow_html=True)
                                    
                                    st.markdown("### 🔍 Specific Examples")
                                    st.markdown(format_as_bullets(feedback.specific_examples), unsafe_allow_html=True)
                                    
                                    st.markdown("### 🎯 Areas for Improvement")
                                    st.markdown(format_as_bullets(feedback.areas_for_improvement), unsafe_allow_html=True)
                                    
                                    st.markdown("### 📝 Additional Comments")
                                    st.markdown(f"<div class='feedback-section'>{feedback.additional_comments}</div>", unsafe_allow_html=True)
                                    
                                    st.markdown("### ❓ Follow-up Questions")
                                    st.markdown(f"<div class='feedback-section'>{feedback.follow_up_questions}</div>", unsafe_allow_html=True)
                                    
                                    # Concerns/Red Flags - Full width if present
                                    if feedback.concerns_red_flags and feedback.concerns_red_flags.strip():
                                        st.markdown("### ⚠️ Concerns/Red Flags")
                                        st.error(feedback.concerns_red_flags)
                                    
                                    # Add some spacing
                                    st.markdown("---")
                                    
                                except Exception as e:
                                    st.error(f"❌ Error analyzing transcript: {str(e)}")
                                    if 'tmp_path' in locals():
                                        try:
                                            os.unlink(tmp_path)
                                        except:
                                            pass
            
            # Batch analysis option
            if len(uploaded_transcripts) > 1:
                st.subheader("⚡ Batch Analysis")
                st.info("🚀 Analyze all transcripts at once with automatic metadata extraction")
                
                if st.button("🚀 Analyze All Transcripts", type="primary"):
                    with st.spinner("Performing batch analysis with automatic metadata extraction..."):
                        try:
                            analyzer = InterviewAnalyzer(api_key, model_choice)
                            batch_feedbacks = []
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, uploaded_file in enumerate(uploaded_transcripts):
                                status_text.text(f"Processing {uploaded_file.name}...")
                                
                                # Save file temporarily
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    tmp_path = tmp_file.name
                                
                                # Analyze with automatic metadata extraction
                                feedback = analyzer.analyze_pdf_transcript(
                                    pdf_path=tmp_path
                                )
                                
                                batch_feedbacks.append(feedback)
                                os.unlink(tmp_path)
                                
                                progress_bar.progress((i + 1) / len(uploaded_transcripts))
                            
                            # Store all feedbacks
                            st.session_state.analyzed_feedbacks.extend(batch_feedbacks)
                            
                            status_text.text("")
                            progress_bar.empty()
                            st.success(f"✅ Batch analysis completed for {len(batch_feedbacks)} transcripts!")
                            
                            # Show summary of extracted metadata in better format
                            st.markdown("### 📋 Batch Analysis Summary")
                            
                            # Create summary cards for each analyzed transcript
                            for idx, feedback in enumerate(batch_feedbacks):
                                with st.container():
                                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                                    
                                    with summary_col1:
                                        st.markdown(f"**📄 File {idx+1}**")
                                        st.markdown(f"👤 **{feedback.candidate_name}**")
                                    
                                    with summary_col2:
                                        st.markdown(f"**💼 Role**")
                                        st.markdown(f"{feedback.role}")
                                    
                                    with summary_col3:
                                        st.markdown(f"**🔄 Round**")
                                        st.markdown(f"{feedback.interview_round}")
                                    
                                    with summary_col4:
                                        st.markdown(f"**📈 Rating**")
                                        rating_color = "🟢" if feedback.overall_rating >= 4 else "🟡" if feedback.overall_rating >= 3 else "🔴"
                                        st.markdown(f"{rating_color} **{feedback.overall_rating}/5**")
                                    
                                    st.markdown("---")
                            
                        except Exception as e:
                            st.error(f"❌ Error during batch analysis: {str(e)}")
    else:
        st.info("📁 Upload PDF transcript files to get started. The AI will automatically extract candidate information from the content.")
    
    # Display all analyzed feedbacks with improved layout
    if st.session_state.analyzed_feedbacks:
        st.subheader("📊 Complete Analysis Dashboard")
        
        # Summary statistics
        feedbacks = st.session_state.analyzed_feedbacks
        
        # Create overview metrics
        if feedbacks:
            overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
            
            avg_overall = sum(f.overall_rating for f in feedbacks) / len(feedbacks)
            avg_technical = sum(f.technical_skills for f in feedbacks) / len(feedbacks)
            avg_communication = sum(f.communication for f in feedbacks) / len(feedbacks)
            thumbs_up_count = sum(1 for f in feedbacks if "Up" in f.thumbs_rating)
            
            with overview_col1:
                st.metric("📊 Total Analyzed", len(feedbacks))
            with overview_col2:
                st.metric("📈 Avg Overall Rating", f"{avg_overall:.1f}/5")
            with overview_col3:
                st.metric("🔧 Avg Technical", f"{avg_technical:.1f}/5")
            with overview_col4:
                st.metric("👍 Thumbs Up", f"{thumbs_up_count}/{len(feedbacks)}")
        
        # Detailed results table
        df_data = []
        for feedback in feedbacks:
            df_data.append({
                'Candidate': feedback.candidate_name,
                'Role': feedback.role,
                'Round': feedback.interview_round,
                'Overall': f"{feedback.overall_rating}/5",
                'Technical': f"{feedback.technical_skills}/5",
                'Communication': f"{feedback.communication}/5",
                'Problem Solving': f"{feedback.problem_solving}/5",
                'Cultural Fit': f"{feedback.cultural_fit}/5",
                'Thumbs': feedback.thumbs_rating,
                'Recommendation': feedback.recommendation,
                'Date': feedback.date
            })
        
        df = pd.DataFrame(df_data)
        
        # Display summary table with better formatting
        st.markdown("### 📋 Detailed Results Table")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export options in better layout
        st.subheader("📥 Export & Actions")
        
        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        
        with export_col1:
            if st.button("📊 Export to Excel", type="primary"):
                try:
                    analyzer = InterviewAnalyzer(api_key if api_key else "dummy", model_choice)
                    
                    # Use the Streamlit-safe method that returns bytes
                    excel_bytes = analyzer.export_feedback_to_excel_streamlit_safe(feedbacks)
                    
                    if excel_bytes:
                        st.download_button(
                            "📥 Download Excel Report",
                            excel_bytes,
                            "interview_feedback_analysis.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_excel"
                        )
                        st.success("✅ Excel report ready for download!")
                    else:
                        # Fallback to CSV
                        data = [asdict(feedback) for feedback in feedbacks]
                        df_export = pd.DataFrame(data)
                        csv_data = df_export.to_csv(index=False, encoding='utf-8-sig')
                        
                        st.download_button(
                            "📥 Download CSV Report (Fallback)",
                            csv_data,
                            "interview_feedback_analysis.csv",
                            "text/csv",
                            key="download_csv_fallback"
                        )
                        st.warning("⚠️ Excel export failed, CSV version ready for download")
                        
                except Exception as e:
                    st.error(f"Error creating export: {str(e)}")
                    # Emergency CSV fallback
                    try:
                        data = [asdict(feedback) for feedback in feedbacks]
                        df_export = pd.DataFrame(data)
                        csv_data = df_export.to_csv(index=False, encoding='utf-8-sig')
                        
                        st.download_button(
                            "📥 Download CSV Report (Emergency)",
                            csv_data,
                            "interview_feedback_analysis.csv",
                            "text/csv",
                            key="download_csv_emergency"
                        )
                        st.info("📄 CSV version available as fallback")
                    except Exception as csv_error:
                        st.error(f"All export methods failed: {csv_error}")
        
        # with export_col2:
        #     if st.button("🎭 Generate Blind Report", type="secondary"):
        #         try:
        #             analyzer = InterviewAnalyzer(api_key if api_key else "dummy", model_choice)
                    
        #             # Use the in-memory method for blind report
        #             blind_bytes = analyzer.generate_blind_feedback_report_bytes(feedbacks)
                    
        #             if blind_bytes:
        #                 st.download_button(
        #                     "📥 Download Blind Report",
        #                     blind_bytes,
        #                     "blind_interview_feedback.xlsx",
        #                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        #                     key="download_blind"
        #                 )
        #                 st.success("✅ Blind report ready for download!")
        #             else:
        #                 # CSV fallback for blind report
        #                 blind_feedback = []
        #                 for i, feedback in enumerate(feedbacks):
        #                     blind_entry = {
        #                         'interview_id': f"INT_{i+1:03d}",
        #                         'round': feedback.interview_round,
        #                         'role': feedback.role,
        #                         'technical_skills': feedback.technical_skills,
        #                         'communication': feedback.communication,
        #                         'problem_solving': feedback.problem_solving, 
        #                         'cultural_fit': feedback.cultural_fit,
        #                         'experience_relevance': feedback.experience_relevance,
        #                         'overall_rating': feedback.overall_rating,
        #                         'thumbs_rating': feedback.thumbs_rating,
        #                         'recommendation': feedback.recommendation,
        #                         'strengths': feedback.strengths,
        #                         'areas_for_improvement': feedback.areas_for_improvement,
        #                         'concerns_red_flags': feedback.concerns_red_flags
        #                     }
        #                     blind_feedback.append(blind_entry)
                        
        #                 df_blind = pd.DataFrame(blind_feedback)
        #                 csv_data = df_blind.to_csv(index=False, encoding='utf-8-sig')
                        
        #                 st.download_button(
        #                     "📥 Download Blind Report (CSV)",
        #                     csv_data,
        #                     "blind_interview_feedback.csv",
        #                     "text/csv",
        #                     key="download_blind_csv"
        #                 )
        #                 st.info("📄 CSV version of blind report available")
                        
        #         except Exception as e:
        #             st.error(f"Error creating blind report: {str(e)}")
        
        # with export_col3:
        #     if st.button("📋 View Summary", type="secondary"):
        #         # Show detailed summary in expander
        #         with st.expander("📊 Detailed Analysis Summary", expanded=True):
        #             for i, feedback in enumerate(feedbacks, 1):
        #                 st.markdown(f"### {i}. {feedback.candidate_name} - {feedback.role}")
                        
        #                 sum_col1, sum_col2, sum_col3 = st.columns(3)
                        
        #                 with sum_col1:
        #                     st.markdown("**📈 Ratings**")
        #                     st.markdown(f"- Overall: {feedback.overall_rating}/5")
        #                     st.markdown(f"- Technical: {feedback.technical_skills}/5")
        #                     st.markdown(f"- Communication: {feedback.communication}/5")
                        
        #                 with sum_col2:
        #                     st.markdown("**🎯 Key Points**")
        #                     st.markdown(f"- Round: {feedback.interview_round}")
        #                     st.markdown(f"- Thumbs: {feedback.thumbs_rating}")
        #                     st.markdown(f"- Recommendation: {feedback.recommendation}")
                        
        #                 with sum_col3:
        #                     st.markdown("**📝 Notes**")
        #                     strengths_preview = feedback.strengths[:100] + "..." if len(feedback.strengths) > 100 else feedback.strengths
        #                     st.markdown(f"Strengths: {strengths_preview}")
                        
        #                 if i < len(feedbacks):
        #                     st.markdown("---")
        
        # with export_col4:
        #     if st.button("🗑️ Clear All Results", type="secondary"):
        #         if st.button("✅ Confirm Clear", key="confirm_clear"):
        #             st.session_state.analyzed_feedbacks = []
        #             st.success("✅ All results cleared!")
        #             st.rerun()
        #         else:
        #             st.warning("Click 'Confirm Clear' to delete all results")

def create_interview_timeline():
    """
    Create a Gantt-style timeline showing interview process milestones.
    Displays email sent date, interview date, and feedback date for each candidate.
    """
    if 'scheduled_interviews' not in st.session_state or not st.session_state.scheduled_interviews:
        st.warning("No scheduled interviews found. Please schedule some interviews first.")
        return None
    
    # Get scheduled interviews and feedback
    scheduled_interviews = st.session_state.scheduled_interviews
    feedback_data = st.session_state.get('analyzed_feedbacks', [])
    
    # Create a mapping of candidate names to their feedback dates
    feedback_dates = {}
    for feedback in feedback_data:
        if hasattr(feedback, 'date') and hasattr(feedback, 'candidate_name'):
            feedback_dates[feedback.candidate_name] = feedback.date
    
    # Create figure
    fig = go.Figure()
    
    # Define colors for different event types
    colors = {
        'email_sent': '#636EFA',    # Blue
        'interview': '#00CC96',     # Green
        'feedback': '#FFA15A'       # Orange
    }
    
    # Add current date line
    current_date = datetime.now().date()
    fig.add_vline(
        x=current_date,
        line_dash="dash",
        line_color="red",
        opacity=0.8,
        name="Today"
    )
    
    # Process each interview
    for interview in scheduled_interviews:
        candidate_name = interview['candidate_name']
        interview_date = datetime.fromisoformat(interview['interview_datetime']).date()
        
        # Estimate email sent date (3 days before interview)
        email_sent_date = interview_date - timedelta(days=3)
        
        # Get feedback date if available
        feedback_date = feedback_dates.get(candidate_name)
        
        # Add email sent marker
        fig.add_trace(go.Scatter(
            x=[email_sent_date],
            y=[candidate_name],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=15,
                color=colors['email_sent'],
                line=dict(width=1, color='white')
            ),
            name='Email Sent',
            legendgroup='Email Sent',
            showlegend=True,
            hoverinfo='text',
            hovertext=f"Email Sent: {email_sent_date.strftime('%Y-%m-%d')}"
        ))
        
        # Add interview date marker
        fig.add_trace(go.Scatter(
            x=[interview_date],
            y=[candidate_name],
            mode='markers',
            marker=dict(
                symbol='square',
                size=15,
                color=colors['interview'],
                line=dict(width=1, color='white')
            ),
            name='Interview',
            legendgroup='Interview',
            showlegend=True,
            hoverinfo='text',
            hovertext=f"Interview: {interview_date.strftime('%Y-%m-%d %H:%M')} - {interview['interview_type']}"
        ))
        
        # Add feedback date marker if available
        if feedback_date:
            try:
                # Try to parse the feedback date if it's a string
                if isinstance(feedback_date, str):
                    feedback_date = datetime.strptime(feedback_date, '%Y-%m-%d').date()
                
                fig.add_trace(go.Scatter(
                    x=[feedback_date],
                    y=[candidate_name],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=15,
                        color=colors['feedback'],
                        line=dict(width=1, color='white')
                    ),
                    name='Feedback Submitted',
                    legendgroup='Feedback',
                    showlegend=True,
                    hoverinfo='text',
                    hovertext=f"Feedback Submitted: {feedback_date.strftime('%Y-%m-%d')}"
                ))
                
                # Add line connecting the events
                fig.add_trace(go.Scatter(
                    x=[email_sent_date, interview_date, feedback_date],
                    y=[candidate_name, candidate_name, candidate_name],
                    mode='lines',
                    line=dict(color='#9e9e9e', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='none'
                ))
                
            except (ValueError, AttributeError) as e:
                print(f"Error processing feedback date for {candidate_name}: {e}")
    
    # Update layout
    fig.update_layout(
        title='Interview Process Timeline',
        title_x=0.5,
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='lightgray',
            tickformat='%Y-%m-%d',
            rangeslider=dict(visible=True)
        ),
        yaxis=dict(
            title='Candidates',
            showgrid=True,
            gridcolor='lightgray',
            autorange='reversed'
        ),
        height=400 + len(scheduled_interviews) * 30,
        margin=dict(l=150, r=20, t=60, b=80, pad=4),
        plot_bgcolor='white',
        hovermode='closest',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    # Add a legend for the current date line
    fig.add_annotation(
        x=current_date,
        y=1.05,
        xref="x",
        yref="paper",
        text="Today",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
        font=dict(color="red")
    )
    
    return fig

def dashboard_tab():
    """Display comprehensive dashboard with visual analytics."""
    st.header("🎨 HumAINex Dashboard")
    st.subheader("Visual Analytics & Insights")
    
    # Check if any data exists
    has_screening = st.session_state.get('top_candidates') is not None
    has_scheduling = bool(st.session_state.get('scheduled_interviews', []))
    has_feedback = bool(st.session_state.get('analyzed_feedbacks', []))
    
    if not any([has_screening, has_scheduling, has_feedback]):
        st.info("🚀 **Welcome to HumAINex Dashboard!**\n\nComplete tasks in other tabs to see comprehensive analytics here:\n- 📋 **Screening**: Screen candidates to see score distributions\n- 📅 **Scheduling**: Schedule interviews to track interview pipeline\n- 🎯 **Feedback**: Analyze feedback to view performance metrics")
        return
    
    # Overview metrics
    st.subheader("🎯 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if has_screening:
            candidates_count = len(st.session_state.top_candidates)
            st.metric("📋 Candidates Screened", candidates_count, delta=f"+{candidates_count}")
        else:
            st.metric("📋 Candidates Screened", "0", help="Complete screening to see data")
    
    with col2:
        if has_scheduling:
            scheduled_count = len(st.session_state.scheduled_interviews)
            st.metric("📅 Interviews Scheduled", scheduled_count, delta=f"+{scheduled_count}")
        else:
            st.metric("📅 Interviews Scheduled", "0", help="Schedule interviews to see data")
    
    with col3:
        if has_feedback:
            feedback_count = len(st.session_state.analyzed_feedbacks)
            st.metric("🎯 Feedback Analyzed", feedback_count, delta=f"+{feedback_count}")
        else:
            st.metric("🎯 Feedback Analyzed", "0", help="Analyze feedback to see data")
    
    with col4:
        # Calculate completion rate
        stages_completed = sum([has_screening, has_scheduling, has_feedback])
        completion_rate = (stages_completed / 3) * 100
        st.metric("✅ Process Completion", f"{completion_rate:.0f}%", delta=f"{stages_completed}/3 stages")
    
    # Display the interview timeline
    st.subheader("📅 Interview Timeline")
    timeline_fig = create_interview_timeline()
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)
    else:
        st.info("No scheduled interviews found. Schedule some interviews to see the timeline.")
    
    # Display the Gantt chart
    st.subheader("📊 Interview Pipeline")
    gantt_fig = create_gantt_chart()
    if gantt_fig:
        st.plotly_chart(gantt_fig, use_container_width=True)
    else:
        st.info("No interview pipeline data available. Screen some candidates to see the pipeline.")
    
    st.subheader("🔄 Process Flow Status")
    
    # Create a flow diagram showing the status of each stage
    flow_col1, flow_col2, flow_col3 = st.columns(3)
    
    with flow_col1:
        if has_screening:
            st.success("✅ **Screening Complete**")
            total_resumes = st.session_state.get('total_resumes_screened', len(st.session_state.top_candidates))
            selected_count = len(st.session_state.top_candidates)
            st.info(f"📊 {selected_count}/{total_resumes} candidates selected")
            if st.session_state.get('screening_completed_at'):
                completed_time = st.session_state.screening_completed_at
                st.caption(f"Completed: {completed_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.warning("⏳ **Screening Pending**")
            st.info("Upload resumes and JD to start")
    
    with flow_col2:
        if has_scheduling:
            st.success("✅ **Scheduling Complete**")
            scheduled_count = len(st.session_state.scheduled_interviews)
            unique_candidates = len(set(interview['candidate_name'] for interview in st.session_state.scheduled_interviews))
            st.info(f"📅 {scheduled_count} interviews for {unique_candidates} candidates")
        elif has_screening:
            st.warning("⏳ **Scheduling Available**")
            st.info("Ready to schedule interviews")
        else:
            st.error("❌ **Screening Required**")
            st.info("Complete screening first")
    
    with flow_col3:
        if has_feedback:
            st.success("✅ **Feedback Analysis Complete**")
            st.info(f"🎯 {len(st.session_state.analyzed_feedbacks)} feedbacks analyzed")
        else:
            st.warning("⏳ **Feedback Analysis Pending**")
            st.info("Upload interview transcripts")
    
    # Screening Analytics
    if has_screening:
        st.markdown("---")
        st.subheader("📋 Screening Analytics")
        
        candidates = st.session_state.top_candidates
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            scores = [candidate.get('evaluation', {}).get('score', 0) for candidate in candidates]
            names = [candidate['name'] for candidate in candidates]
            
            fig_scores = px.bar(
                x=names,
                y=scores,
                title="📈 Candidate Screening Scores",
                labels={'x': 'Candidates', 'y': 'Score (out of 100)'},
                color=scores,
                color_continuous_scale=['red', 'yellow', 'green']
            )
            fig_scores.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_scores, use_container_width=True)
        
        with col2:
            # Resume screening funnel chart
            total_resumes = st.session_state.get('total_resumes_screened', len(candidates))
            selected_candidates = len(candidates)
            rejected_candidates = total_resumes - selected_candidates
            
            # Create pie chart showing screening results
            screening_data = {
                'Selected for Interview': selected_candidates,
                'Not Selected': rejected_candidates
            }
            
            colors = ['#2ecc71', '#e74c3c']  # Green for selected, red for not selected
            
            # fig_screening_funnel = px.pie(
            #     values=list(screening_data.values()),
            #     names=list(screening_data.keys()),
            #     title=f"📊 Screening Results ({total_resumes} Total Resumes)",
            #     color_discrete_sequence=colors
            # )
            # fig_screening_funnel.update_traces(textposition='inside', textinfo='percent+label+value')
            # fig_screening_funnel.update_layout(height=400)
            # st.plotly_chart(fig_screening_funnel, use_container_width=True)
        
        # # Show screening statistics
        # st.markdown("### 📈 Screening Statistics")
        # stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        # with stats_col1:
        #     st.metric("📄 Total Resumes", total_resumes)
        # with stats_col2:
        #     st.metric("✅ Selected", selected_candidates)
        # with stats_col3:
        #     st.metric("❌ Rejected", rejected_candidates)
        # with stats_col4:
        #     selection_rate = (selected_candidates / total_resumes * 100) if total_resumes > 0 else 0
        #     st.metric("📊 Selection Rate", f"{selection_rate:.1f}%")
    
    # Scheduling Analytics
    if has_scheduling:
        interviews = st.session_state.scheduled_interviews
        
        # Show scheduling completion status
        if has_screening:
            total_candidates = len(st.session_state.top_candidates)
            scheduled_candidates = len(set(interview['candidate_name'] for interview in interviews))
            pending_candidates = total_candidates - scheduled_candidates
            st.markdown("---")
            st.markdown("### 📊 Scheduling Progress")
            progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)
            
            with progress_col1:
                st.metric("👥 Total Candidates", total_candidates)
            with progress_col2:
                st.metric("✅ Scheduled", scheduled_candidates)
            with progress_col3:
                st.metric("⏳ Pending", pending_candidates)
            with progress_col4:
                schedule_rate = (scheduled_candidates / total_candidates * 100) if total_candidates > 0 else 0
                st.metric("📈 Schedule Rate", f"{schedule_rate:.1f}%")
        
        # Upcoming interviews
        st.markdown("### 🗓️ Upcoming Interviews")
        
        now = datetime.now()
        upcoming = []
        past = []
        
        for interview in interviews:
            interview_dt = datetime.fromisoformat(interview['interview_datetime'])
            if interview_dt > now:
                upcoming.append((interview, interview_dt))
            else:
                past.append((interview, interview_dt))
        
        upcoming.sort(key=lambda x: x[1])
        
        if upcoming:
            for interview, interview_dt in upcoming[:5]:  # Show next 5
                days_until = (interview_dt.date() - now.date()).days
                time_str = interview_dt.strftime("%I:%M %p")
                date_str = interview_dt.strftime("%B %d, %Y")
                
                if days_until == 0:
                    urgency = "🔴 Today"
                elif days_until == 1:
                    urgency = "🟡 Tomorrow"
                elif days_until <= 3:
                    urgency = f"🟠 In {days_until} days"
                else:
                    urgency = f"🟢 In {days_until} days"
                
                st.info(f"{urgency} | **{interview['candidate_name']}** | {date_str} at {time_str} | {interview['interview_type']}")
        else:
            st.info("No upcoming interviews scheduled")
    
    # Feedback Analytics
    if has_feedback:
        # st.subheader("🎯 Feedback Analytics")
        
        feedbacks = st.session_state.analyzed_feedbacks
        
        # # Create comprehensive analytics
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     # Overall ratings distribution
        #     overall_ratings = [f.overall_rating for f in feedbacks]
            
        #     fig_ratings = px.histogram(
        #         x=overall_ratings,
        #         nbins=5,
        #         title="📈 Overall Ratings Distribution",
        #         labels={'x': 'Rating (1-5)', 'y': 'Number of Candidates'},
        #         color_discrete_sequence=['#1f77b4']
        #     )
        #     fig_ratings.update_layout(height=400)
        #     st.plotly_chart(fig_ratings, use_container_width=True)
        
        # with col2:
        #     # Thumbs rating distribution
        #     thumbs_ratings = [f.thumbs_rating for f in feedbacks]
        #     thumbs_counts = pd.Series(thumbs_ratings).value_counts()
            
        #     colors_map = {}
        #     for rating in thumbs_counts.index:
        #         if "Up" in rating:
        #             colors_map[rating] = '#2ecc71'
        #         elif "Down" in rating:
        #             colors_map[rating] = '#e74c3c'
        #         else:
        #             colors_map[rating] = '#f39c12'
            
        #     fig_thumbs = px.pie(
        #         values=thumbs_counts.values,
        #         names=thumbs_counts.index,
        #         title="👍 Thumbs Rating Distribution",
        #         color=thumbs_counts.index,
        #         color_discrete_map=colors_map
        #     )
        #     fig_thumbs.update_layout(height=400)
        #     st.plotly_chart(fig_thumbs, use_container_width=True)
        
        # Top performers from feedback
        st.markdown("### 🌟 Top Performers (Based on Feedback)")
        
        sorted_feedback = sorted(feedbacks, key=lambda x: x.overall_rating, reverse=True)
        
        for i, feedback in enumerate(sorted_feedback[:3], 1):
            with st.container():
                rank_col, info_col, metrics_col, status_col = st.columns([1, 2, 3, 2])
                
                with rank_col:
                    medal = "🏆" if i == 1 else "🥈" if i == 2 else "🥉"
                    st.markdown(f"## {medal}")
                
                with info_col:
                    st.markdown(f"**{feedback.candidate_name}**")
                    st.markdown(f"Role: {feedback.role}")
                    st.markdown(f"Round: {feedback.interview_round}")
                
                with metrics_col:
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Overall", f"{feedback.overall_rating}/5")
                        st.metric("Technical", f"{feedback.technical_skills}/5")
                    with metric_col2:
                        st.metric("Communication", f"{feedback.communication}/5")
                        st.metric("Problem Solving", f"{feedback.problem_solving}/5")
                
                with status_col:
                    thumbs_emoji = "👍" if "Up" in feedback.thumbs_rating else "👎" if "Down" in feedback.thumbs_rating else "🤚"
                    st.markdown(f"**{thumbs_emoji} {feedback.thumbs_rating}**")
                    
                    if "hire" in feedback.recommendation.lower() and "no" not in feedback.recommendation.lower():
                        st.success("✅ Recommended")
                    elif "not" in feedback.recommendation.lower() or "no hire" in feedback.recommendation.lower():
                        st.error("❌ Not Recommended")
                    else:
                        st.warning("⚠️ Under Review")
                
                st.markdown("---")
    
    # Action recommendations
    # st.subheader("💡 Recommended Actions")
    
    # actions = []
    
    # if not has_screening:
    #     actions.append("📋 **Start Screening**: Upload job description and resumes in the Screening tab")
    # elif has_screening and not has_scheduling:
    #     actions.append("📅 **Schedule Interviews**: Move to the Scheduling tab to set up interviews with screened candidates")
    # elif has_scheduling and not has_feedback:
    #     actions.append("🎯 **Analyze Feedback**: Upload interview transcripts in the Feedback Analysis tab")
    
    # if has_feedback:
    #     avg_rating = sum(f.overall_rating for f in st.session_state.analyzed_feedbacks) / len(st.session_state.analyzed_feedbacks)
    #     if avg_rating < 3.0:
    #         actions.append("⚠️ **Review Process**: Average ratings are low, consider reviewing screening criteria")
    #     else:
    #         actions.append("🎉 **Process Complete**: All stages completed successfully!")
    
    # for action in actions:
    #     st.info(action)
    
    # if not actions:
    #     st.success("🎉 **All Done!** Your hiring process is complete. Use the export features in each tab to generate reports.")

def interviewer_dashboard():
    """Special dashboard for interviewer role - shows only feedback analysis content."""
    st.header("🎯 Interviewer Dashboard")
    st.subheader("Interview Feedback Analysis")
    
    # Show user info
    st.info(f"👤 **Logged in as:** {st.session_state.username} (Interviewer)")
    
    # Call the feedback analysis functionality
    feedback_analysis_tab()

def main():
     # Check authentication first
    is_authenticated, user_role = check_authentication()
    
    if not is_authenticated:
        login_page()
        return
    
    # User is authenticated - show main app
    st.title("🤝 HumAINex")
    st.subheader("Your AI-powered Candidate Interview Tracking System")
    
    # Show user info and logout in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 👤 User Info")
        st.markdown(f"**Logged in as:** {st.session_state.username}")
        st.markdown(f"**Role:** {user_role.upper()}")
        
        if st.button("🚪 Logout", type="secondary"):
            logout()
        
        # Show user management for HR users
        if user_role == 'hr':
            create_user_management_section()
    
    
    # Initialize session state
    if 'top_candidates' not in st.session_state:
        st.session_state.top_candidates = None
    if 'scheduled_interviews' not in st.session_state:
        st.session_state.scheduled_interviews = []
    if 'analyzed_feedbacks' not in st.session_state:
        st.session_state.analyzed_feedbacks = []
    if 'total_resumes_screened' not in st.session_state:
        st.session_state.total_resumes_screened = 0
    if 'selected_candidates_count' not in st.session_state:
        st.session_state.selected_candidates_count = 0
    
    # Role-based access control
    if user_role == 'hr':
        # HR users get full access to all tabs
        
        # Check if user management should be shown
        if st.session_state.get('show_user_management', False):
            display_user_management()
            if st.button("🔙 Back to Main App"):
                st.session_state.show_user_management = False
                st.rerun()
            return
        
        # Create tabs for HR users
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Screening", "📅 Schedule Interviews", "🎯 Feedback Analysis", "📊 Dashboard"])
        
        with tab1:
            screening_tab()
        
        with tab2:
            scheduling_tab()
        
        with tab3:
            feedback_analysis_tab()
        
        with tab4:
            dashboard_tab()
            
    elif user_role == 'interviewer':
        # Interviewer users only get access to feedback analysis
        interviewer_dashboard()
    
    else:
        # Default/unknown role - limited access
        st.warning("⚠️ Your account has limited access. Contact your administrator for more permissions.")
        st.info("Available features: Feedback Analysis")
        feedback_analysis_tab()

if __name__ == "__main__":
    main()